"""S3-compatible media store with handrolled AWS SigV4 signing.

Targets AWS S3, Cloudflare R2, MinIO, and other S3-compatible endpoints.
Implements path-style URLs (`<endpoint>/<bucket>/<key>`) — the lowest common
denominator across providers. Only PUT / GET / HEAD are implemented;
multipart, lifecycle, and listing operations are out of scope (any blob we
store is sub-5GB by construction so multipart is unnecessary).

The SigV4 implementation follows the AWS reference algorithm but is local
to this module to keep `pydantic-ai-harness` free of `botocore` / `boto3`.
If a provider needs deeper integration, write a subclass that overrides
`_request`.
"""

from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from urllib.parse import quote

from pydantic_ai_harness.media._store import media_uri_for, parse_media_uri

if TYPE_CHECKING:
    import httpx


_ALGORITHM = 'AWS4-HMAC-SHA256'
_SERVICE = 's3'


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _hex_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac_sha256(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()


def _canonical_uri(path: str) -> str:
    """Percent-encode each path segment but keep `/` separators.

    AWS canonicalization requires double-encoding for non-S3 services but
    single-encoding for S3 — we use single-encoding throughout.
    """
    return '/' + '/'.join(quote(segment, safe='') for segment in path.lstrip('/').split('/'))


def _signing_key(secret_access_key: str, date_stamp: str, region: str) -> bytes:
    k_date = _hmac_sha256(f'AWS4{secret_access_key}'.encode(), date_stamp)
    k_region = _hmac_sha256(k_date, region)
    k_service = _hmac_sha256(k_region, _SERVICE)
    return _hmac_sha256(k_service, 'aws4_request')


def sign_request(
    *,
    method: str,
    host: str,
    path: str,
    body: bytes,
    region: str,
    access_key_id: str,
    secret_access_key: str,
    content_type: str | None = None,
    now: datetime | None = None,
) -> dict[str, str]:
    """Return the headers (including `Authorization`) to send with an S3 request.

    `path` should start with `/` and already include any bucket prefix —
    e.g. `/my-bucket/object-key`. Query strings are not supported in v1
    (we never send them for PUT/GET/HEAD of a single object).
    """
    timestamp = now or _utcnow()
    amz_date = timestamp.strftime('%Y%m%dT%H%M%SZ')
    date_stamp = timestamp.strftime('%Y%m%d')

    payload_hash = _hex_sha256(body)
    headers_to_sign: dict[str, str] = {
        'host': host,
        'x-amz-content-sha256': payload_hash,
        'x-amz-date': amz_date,
    }
    if content_type is not None:
        headers_to_sign['content-type'] = content_type

    sorted_keys = sorted(headers_to_sign)
    canonical_headers = ''.join(f'{k}:{headers_to_sign[k]}\n' for k in sorted_keys)
    signed_headers = ';'.join(sorted_keys)

    canonical_request = '\n'.join(
        [
            method.upper(),
            _canonical_uri(path),
            '',
            canonical_headers,
            signed_headers,
            payload_hash,
        ]
    )

    credential_scope = f'{date_stamp}/{region}/{_SERVICE}/aws4_request'
    string_to_sign = '\n'.join(
        [
            _ALGORITHM,
            amz_date,
            credential_scope,
            _hex_sha256(canonical_request.encode('utf-8')),
        ]
    )

    signature = hmac.new(
        _signing_key(secret_access_key, date_stamp, region),
        string_to_sign.encode('utf-8'),
        hashlib.sha256,
    ).hexdigest()

    authorization = (
        f'{_ALGORITHM} '
        f'Credential={access_key_id}/{credential_scope}, '
        f'SignedHeaders={signed_headers}, '
        f'Signature={signature}'
    )
    return {**headers_to_sign, 'authorization': authorization}


class S3MediaStore:
    """S3-compatible content-addressed store using path-style URLs + SigV4.

    Works with AWS S3 (`endpoint='https://s3.<region>.amazonaws.com'`),
    Cloudflare R2 (`endpoint='https://<account_id>.r2.cloudflarestorage.com'`,
    `region='auto'`), MinIO, and other compatible providers.

    Keys are derived from the content hash — caller never picks them. The
    object key in the bucket is `<key_prefix><sha256>.bin`.

    Pass an existing `httpx.AsyncClient` via `client=` to share connection
    pooling with the rest of your app; otherwise a fresh client is opened
    per call. Owning-vs-borrowing follows the usual httpx convention.
    """

    def __init__(
        self,
        *,
        bucket: str,
        endpoint: str,
        region: str,
        access_key_id: str,
        secret_access_key: str,
        key_prefix: str = '',
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._bucket = bucket
        self._endpoint = endpoint.rstrip('/')
        self._region = region
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._key_prefix = key_prefix
        self._client = client

    def _object_path(self, digest: str) -> str:
        return f'/{self._bucket}/{self._key_prefix}{digest}.bin'

    def _host(self) -> str:
        host = self._endpoint.split('://', 1)[1]
        return host.split('/', 1)[0]

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes = b'',
        content_type: str | None = None,
    ) -> httpx.Response:
        import httpx

        headers = sign_request(
            method=method,
            host=self._host(),
            path=path,
            body=body,
            region=self._region,
            access_key_id=self._access_key_id,
            secret_access_key=self._secret_access_key,
            content_type=content_type,
        )
        url = f'{self._endpoint}{path}'
        if self._client is not None:
            return await self._client.request(method, url, content=body, headers=headers)
        async with httpx.AsyncClient() as client:
            return await client.request(method, url, content=body, headers=headers)

    async def put(self, data: bytes, *, media_type: str | None = None) -> str:
        uri = media_uri_for(data)
        digest = parse_media_uri(uri)
        response = await self._request(
            'PUT',
            self._object_path(digest),
            body=data,
            content_type=media_type,
        )
        if response.status_code // 100 != 2:
            raise RuntimeError(f'S3 PUT failed for {digest}: {response.status_code} {response.text[:200]}')
        return uri

    async def get(self, uri: str) -> bytes:
        digest = parse_media_uri(uri)
        response = await self._request('GET', self._object_path(digest))
        if response.status_code == 404:
            raise FileNotFoundError(f'media not found: {digest}')
        if response.status_code // 100 != 2:
            raise RuntimeError(f'S3 GET failed for {digest}: {response.status_code} {response.text[:200]}')
        return response.content

    async def exists(self, uri: str) -> bool:
        digest = parse_media_uri(uri)
        response = await self._request('HEAD', self._object_path(digest))
        if response.status_code == 404:
            return False
        if response.status_code // 100 != 2:
            raise RuntimeError(f'S3 HEAD failed for {digest}: {response.status_code} {response.text[:200]}')
        return True
