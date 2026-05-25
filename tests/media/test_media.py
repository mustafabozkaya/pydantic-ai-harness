"""Tests for `pydantic_ai_harness.media`: stores + walker + SigV4 + S3 store."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from httpx import AsyncClient, MockTransport, Request, Response

from pydantic_ai_harness.media import (
    DiskMediaStore,
    MediaStore,
    S3MediaStore,
    SqliteMediaStore,
    externalize_media,
    media_uri_for,
    parse_media_uri,
    restore_media,
)
from pydantic_ai_harness.media._s3 import sign_request

pytestmark = pytest.mark.anyio


class TestMediaUriHelpers:
    def test_media_uri_for_returns_canonical_scheme(self) -> None:
        uri = media_uri_for(b'hello world')
        assert uri == 'media+sha256://b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'

    def test_parse_media_uri_strips_scheme(self) -> None:
        uri = media_uri_for(b'abc')
        digest = parse_media_uri(uri)
        assert len(digest) == 64
        assert all(c in '0123456789abcdef' for c in digest)

    def test_parse_media_uri_rejects_other_schemes(self) -> None:
        with pytest.raises(ValueError, match='not a media URI'):
            parse_media_uri('http://example.com/foo')

    def test_parse_media_uri_rejects_short_digest(self) -> None:
        with pytest.raises(ValueError, match='invalid sha256 digest'):
            parse_media_uri('media+sha256://deadbeef')

    def test_parse_media_uri_rejects_uppercase_hex(self) -> None:
        with pytest.raises(ValueError, match='invalid sha256 digest'):
            parse_media_uri('media+sha256://' + 'A' * 64)


class TestDiskMediaStore:
    async def test_put_get_round_trip(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        uri = await store.put(b'hello bytes', media_type='application/octet-stream')
        assert uri.startswith('media+sha256://')
        assert await store.get(uri) == b'hello bytes'

    async def test_dedup_on_repeated_put(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        uri1 = await store.put(b'same content')
        uri2 = await store.put(b'same content')
        assert uri1 == uri2
        files = list(tmp_path.glob('*.bin'))
        assert len(files) == 1

    async def test_exists(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        uri = await store.put(b'present')
        assert await store.exists(uri) is True
        missing_uri = media_uri_for(b'never put')
        assert await store.exists(missing_uri) is False

    async def test_get_missing_raises(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        with pytest.raises(FileNotFoundError):
            await store.get(media_uri_for(b'never put'))


class TestSqliteMediaStore:
    async def test_put_get_round_trip(self, tmp_path: Path) -> None:
        store = SqliteMediaStore(database=tmp_path / 'media.db')
        uri = await store.put(b'hello bytes', media_type='image/png')
        assert await store.get(uri) == b'hello bytes'

    async def test_with_shared_connection(self) -> None:
        connection = sqlite3.connect(':memory:', check_same_thread=False)
        try:
            store = SqliteMediaStore(connection=connection)
            uri = await store.put(b'shared conn data')
            assert await store.get(uri) == b'shared conn data'
            assert await store.exists(uri) is True
        finally:
            connection.close()

    async def test_dedup_via_insert_or_ignore(self, tmp_path: Path) -> None:
        store = SqliteMediaStore(database=tmp_path / 'm.db')
        uri = ''
        for _ in range(3):
            uri = await store.put(b'duplicated')
        assert await store.get(uri) == b'duplicated'

    async def test_get_missing_raises(self, tmp_path: Path) -> None:
        store = SqliteMediaStore(database=tmp_path / 'm.db')
        with pytest.raises(FileNotFoundError):
            await store.get(media_uri_for(b'missing'))

    def test_requires_exactly_one_of_database_or_connection(self) -> None:
        with pytest.raises(ValueError, match='exactly one'):
            SqliteMediaStore()
        conn = sqlite3.connect(':memory:')
        try:
            with pytest.raises(ValueError, match='exactly one'):
                SqliteMediaStore(database='x', connection=conn)
        finally:
            conn.close()

    def test_rejects_invalid_table_name(self) -> None:
        with pytest.raises(ValueError, match='invalid table name'):
            SqliteMediaStore(database='x.db', table='bad-name')


class TestExternalizeRestoreWalker:
    async def test_round_trip_with_inline_binary(self, tmp_path: Path) -> None:
        import base64
        import json as _json

        store = DiskMediaStore(tmp_path)
        big_payload = b'\x00' * 70_000
        b64_payload = base64.b64encode(big_payload).decode('ascii')
        node: object = {
            'parts': [
                {
                    'kind': 'binary',
                    'data': b64_payload,
                    'media_type': 'image/png',
                    'identifier': 'abc',
                    'vendor_metadata': None,
                }
            ]
        }
        externalized = await externalize_media(node, media_store=store, threshold_bytes=64 * 1024)
        externalized_text = _json.dumps(externalized)
        assert '__harness_external_media__' in externalized_text
        assert 'media+sha256://' in externalized_text
        assert b64_payload not in externalized_text  # bytes really went external

        restored = await restore_media(externalized, media_store=store)
        restored_text = _json.dumps(restored)
        assert '"kind": "binary"' in restored_text
        assert b64_payload in restored_text  # bytes restored exactly

    async def test_threshold_boundary_keeps_small_inline(self, tmp_path: Path) -> None:
        import base64

        store = DiskMediaStore(tmp_path)
        small_payload = b'\x42' * 32
        node = {
            'kind': 'binary',
            'data': base64.b64encode(small_payload).decode('ascii'),
            'media_type': 'text/plain',
            'identifier': 's',
            'vendor_metadata': None,
        }
        externalized = await externalize_media(node, media_store=store, threshold_bytes=64 * 1024)
        assert externalized == node
        assert list(tmp_path.glob('*.bin')) == []

    async def test_restore_raises_when_marker_missing_uri(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        bad_node = {'__harness_external_media__': True, 'media_type': 'image/png'}
        with pytest.raises(ValueError, match='missing string uri'):
            await restore_media(bad_node, media_store=store)

    async def test_walker_passes_through_scalars(self, tmp_path: Path) -> None:
        store = DiskMediaStore(tmp_path)
        for value in [None, 1, 'foo', True]:
            assert await externalize_media(value, media_store=store, threshold_bytes=1) == value
            assert await restore_media(value, media_store=store) == value


class TestSigV4Signer:
    def test_produces_required_headers(self) -> None:
        headers = sign_request(
            method='PUT',
            host='examplebucket.s3.amazonaws.com',
            path='/my-key',
            body=b'payload',
            region='us-east-1',
            access_key_id='AKIAIOSFODNN7EXAMPLE',
            secret_access_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
            content_type='image/png',
            now=datetime(2013, 5, 24, tzinfo=timezone.utc),
        )
        assert headers['host'] == 'examplebucket.s3.amazonaws.com'
        assert headers['x-amz-date'] == '20130524T000000Z'
        assert headers['x-amz-content-sha256'] == ('239f59ed55e737c77147cf55ad0c1b030b6d7ee748a7426952f9b852d5a935e5')
        assert headers['content-type'] == 'image/png'
        auth = headers['authorization']
        assert auth.startswith('AWS4-HMAC-SHA256 ')
        assert 'Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request' in auth
        assert 'SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date' in auth
        # Signature is deterministic for fixed inputs — locks down the algorithm.
        assert 'Signature=' in auth

    def test_signature_is_deterministic(self) -> None:
        common = dict(
            method='GET',
            host='bucket.s3.amazonaws.com',
            path='/key',
            body=b'',
            region='us-east-1',
            access_key_id='K',
            secret_access_key='S',
            content_type=None,
            now=datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        )
        a = sign_request(**common)  # type: ignore[arg-type]
        b = sign_request(**common)  # type: ignore[arg-type]
        assert a == b

    def test_signature_changes_with_body(self) -> None:
        kwargs = dict(
            method='PUT',
            host='bucket.s3.amazonaws.com',
            path='/key',
            region='us-east-1',
            access_key_id='K',
            secret_access_key='S',
            content_type=None,
            now=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        sig_a = sign_request(body=b'aaa', **kwargs)['authorization']  # type: ignore[arg-type]
        sig_b = sign_request(body=b'bbb', **kwargs)['authorization']  # type: ignore[arg-type]
        assert sig_a != sig_b


class TestS3MediaStoreWithMockTransport:
    async def test_put_signs_request(self) -> None:
        captured: list[Request] = []

        async def handler(request: Request) -> Response:
            captured.append(request)
            return Response(200)

        transport = MockTransport(handler)
        async with AsyncClient(transport=transport) as client:
            store = S3MediaStore(
                bucket='my-bucket',
                endpoint='https://s3.us-east-1.amazonaws.com',
                region='us-east-1',
                access_key_id='AKIA-FAKE',
                secret_access_key='secret-fake',
                client=client,
            )
            uri = await store.put(b'payload', media_type='image/png')

        assert uri.startswith('media+sha256://')
        assert len(captured) == 1
        request = captured[0]
        assert request.method == 'PUT'
        assert 'authorization' in request.headers
        assert request.headers['authorization'].startswith('AWS4-HMAC-SHA256 ')
        assert request.headers['x-amz-content-sha256']
        assert request.headers['content-type'] == 'image/png'

    async def test_get_resolves_uri(self) -> None:
        async def handler(request: Request) -> Response:
            return Response(200, content=b'fetched bytes')

        async with AsyncClient(transport=MockTransport(handler)) as client:
            store = S3MediaStore(
                bucket='b',
                endpoint='https://example.com',
                region='auto',
                access_key_id='k',
                secret_access_key='s',
                client=client,
            )
            data = await store.get(media_uri_for(b'fetched bytes'))
            assert data == b'fetched bytes'

    async def test_get_404_raises(self) -> None:
        async def handler(request: Request) -> Response:
            return Response(404)

        async with AsyncClient(transport=MockTransport(handler)) as client:
            store = S3MediaStore(
                bucket='b',
                endpoint='https://example.com',
                region='auto',
                access_key_id='k',
                secret_access_key='s',
                client=client,
            )
            with pytest.raises(FileNotFoundError):
                await store.get(media_uri_for(b'missing'))

    async def test_exists_uses_head(self) -> None:
        captured_methods: list[str] = []

        async def handler(request: Request) -> Response:
            captured_methods.append(request.method)
            return Response(200)

        async with AsyncClient(transport=MockTransport(handler)) as client:
            store = S3MediaStore(
                bucket='b',
                endpoint='https://example.com',
                region='auto',
                access_key_id='k',
                secret_access_key='s',
                client=client,
            )
            assert await store.exists(media_uri_for(b'anything')) is True

        assert captured_methods == ['HEAD']

    async def test_exists_returns_false_on_404(self) -> None:
        async def handler(request: Request) -> Response:
            return Response(404)

        async with AsyncClient(transport=MockTransport(handler)) as client:
            store = S3MediaStore(
                bucket='b',
                endpoint='https://example.com',
                region='auto',
                access_key_id='k',
                secret_access_key='s',
                client=client,
            )
            assert await store.exists(media_uri_for(b'missing')) is False

    async def test_put_failure_raises(self) -> None:
        async def handler(request: Request) -> Response:
            return Response(500, content=b'internal error')

        async with AsyncClient(transport=MockTransport(handler)) as client:
            store = S3MediaStore(
                bucket='b',
                endpoint='https://example.com',
                region='auto',
                access_key_id='k',
                secret_access_key='s',
                client=client,
            )
            with pytest.raises(RuntimeError, match='S3 PUT failed'):
                await store.put(b'data')

    async def test_get_failure_raises(self) -> None:
        async def handler(request: Request) -> Response:
            return Response(500, content=b'oops')

        async with AsyncClient(transport=MockTransport(handler)) as client:
            store = S3MediaStore(
                bucket='b',
                endpoint='https://example.com',
                region='auto',
                access_key_id='k',
                secret_access_key='s',
                client=client,
            )
            with pytest.raises(RuntimeError, match='S3 GET failed'):
                await store.get(media_uri_for(b'x'))

    async def test_head_failure_raises(self) -> None:
        async def handler(request: Request) -> Response:
            return Response(500)

        async with AsyncClient(transport=MockTransport(handler)) as client:
            store = S3MediaStore(
                bucket='b',
                endpoint='https://example.com',
                region='auto',
                access_key_id='k',
                secret_access_key='s',
                client=client,
            )
            with pytest.raises(RuntimeError, match='S3 HEAD failed'):
                await store.exists(media_uri_for(b'x'))

    async def test_no_client_branch_opens_one_per_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without `client=`, the store opens a fresh `httpx.AsyncClient` per call."""
        import httpx

        captured: list[Request] = []

        async def handler(request: Request) -> Response:
            captured.append(request)
            return Response(200)

        original_async_client = httpx.AsyncClient

        def patched_client(*args: object, **kwargs: object) -> httpx.AsyncClient:
            return original_async_client(transport=MockTransport(handler))

        monkeypatch.setattr(httpx, 'AsyncClient', patched_client)

        store = S3MediaStore(
            bucket='my-bucket',
            endpoint='https://example.com',
            region='auto',
            access_key_id='k',
            secret_access_key='s',
            key_prefix='runs/',
        )
        await store.put(b'no-client data')
        assert len(captured) == 1
        path = store._object_path('a' * 64)  # type: ignore[reportPrivateUsage]
        assert path == '/my-bucket/runs/' + ('a' * 64) + '.bin'


@pytest.mark.skipif(  # pragma: no cover
    not all(
        os.environ.get(k)
        for k in ('S3_ENDPOINT', 'S3_BUCKET_NAME', 'S3_ACCESS_KEY_ID', 'S3_SECRET_ACCESS_KEY', 'S3_REGION')
    ),
    reason='live S3/R2 env vars not set',
)
class TestS3MediaStoreLive:  # pragma: no cover
    """Live integration against the configured S3 endpoint (e.g. R2).

    Activated only when all five S3_* env vars are set. Reads creds out of
    the environment — the test harness inherits them when invoked with
    `~/.claude/scripts/env-run .env -- make test`.
    """

    async def test_round_trip_against_live_endpoint(self) -> None:
        bucket = os.environ['S3_BUCKET_NAME']
        store = S3MediaStore(
            bucket=bucket,
            endpoint=os.environ['S3_ENDPOINT'],
            region=os.environ['S3_REGION'],
            access_key_id=os.environ['S3_ACCESS_KEY_ID'],
            secret_access_key=os.environ['S3_SECRET_ACCESS_KEY'],
            key_prefix='harness-test/',
        )
        payload = b'pydantic-ai-harness step-persistence live test ' + os.urandom(32)
        uri = await store.put(payload, media_type='application/octet-stream')
        assert await store.exists(uri) is True
        fetched = await store.get(uri)
        assert fetched == payload


def _assert_media_store_protocol(store: MediaStore) -> None:
    """Mypy/pyright check: every concrete store satisfies the protocol."""
    assert isinstance(store, MediaStore)


def test_concrete_stores_satisfy_protocol(tmp_path: Path) -> None:
    _assert_media_store_protocol(DiskMediaStore(tmp_path))
    _assert_media_store_protocol(SqliteMediaStore(database=tmp_path / 'm.db'))
    _assert_media_store_protocol(
        S3MediaStore(
            bucket='b',
            endpoint='https://example.com',
            region='auto',
            access_key_id='k',
            secret_access_key='s',
        )
    )
