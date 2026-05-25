"""Local `MediaStore` protocol + `DiskMediaStore` / `SqliteMediaStore`.

The shared URI scheme is `media+sha256://<lowercase-hex>` — content-addressed,
so the same blob written through any store resolves the same way and dedup
is automatic.
"""

from __future__ import annotations

import hashlib
import inspect
import re
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

import anyio.to_thread

_URI_SCHEME = 'media+sha256://'
_HEX_RE = re.compile(r'^[0-9a-f]{64}$')

PublicUrlResolver = Callable[[str], 'str | None | Awaitable[str | None]']
"""User-supplied callable that turns a `media+sha256://<hex>` URI into a fetchable URL.

Sync or async; the store auto-detects via `inspect.isawaitable` on the result.
Return `None` to signal "no URL for this URI" (the resolver may be content-aware
and choose to inline some payloads).

Common shapes:

- **Static base URL** (public bucket, CDN): `lambda uri: f'{base}/{path_from(uri)}'`.
  See `make_static_public_url` for the boilerplate-free version.
- **Presigned URL** (private bucket, rotating signature): async callable that
  signs fresh on each invocation, with TTL captured in the closure.
- **No URL available**: pass nothing — the store's `public_url(uri)` returns `None`.
"""


async def _resolve_public_url(resolver: PublicUrlResolver | None, uri: str) -> str | None:
    if resolver is None:
        return None
    result = resolver(uri)
    if inspect.isawaitable(result):
        return await result
    return result


def make_static_public_url(
    base: str,
    *,
    key_prefix: str = '',
    extension: str = '.bin',
) -> Callable[[str], str]:
    """Build a static-URL resolver for the common "public bucket / CDN" case.

    The returned callable takes a `media+sha256://<hex>` URI and returns
    `{base}/{key_prefix}<hex>{extension}`. Use this when the bytes live at a
    known stable URL — e.g. an R2 public bucket (`https://pub-xxx.r2.dev`)
    or a CDN in front of S3. For presigned URLs or any logic that runs per
    request, pass your own callable instead.
    """
    base = base.rstrip('/')

    def resolver(uri: str) -> str:
        digest = parse_media_uri(uri)
        return f'{base}/{key_prefix}{digest}{extension}'

    return resolver


def media_uri_for(data: bytes) -> str:
    """Return the canonical `media+sha256://<hex>` URI for `data`.

    The URI is the same regardless of which store the bytes are written to —
    content-addressed so two stores holding the same bytes can be queried
    interchangeably.
    """
    return f'{_URI_SCHEME}{hashlib.sha256(data).hexdigest()}'


def parse_media_uri(uri: str) -> str:
    """Return the lowercase hex digest from a `media+sha256://<hex>` URI.

    Raises `ValueError` if `uri` does not match the scheme or the digest is
    not 64 lowercase hex characters.
    """
    if not uri.startswith(_URI_SCHEME):
        raise ValueError(f'not a media URI: {uri!r}')
    digest = uri[len(_URI_SCHEME) :]
    if not _HEX_RE.fullmatch(digest):
        raise ValueError(f'invalid sha256 digest in {uri!r}')
    return digest


@runtime_checkable
class MediaStore(Protocol):
    """Async content-addressed bytes store.

    `put` returns the canonical URI (`media+sha256://<hex>`) for the bytes —
    callers do not pick the key. Implementations may dedup on the hash.

    `media_type` is advisory metadata (e.g. `image/png`); stores are free to
    persist or ignore it. The hash never depends on it.

    `public_url(uri)` returns a URL the model can fetch directly, or `None`
    if the store can't or hasn't been configured to produce one. The forthcoming
    `MediaExternalizer` capability uses this to rewrite `BinaryContent` parts
    to `ImageUrl`/`AudioUrl`/etc. before the model sees the message. Both
    static (CDN, public bucket) and dynamic (presigned, rotating signature)
    URLs fit the same shape — the concrete stores accept a user-supplied
    `PublicUrlResolver` callable.
    """

    async def put(self, data: bytes, *, media_type: str | None = None) -> str: ...  # pragma: no cover

    async def get(self, uri: str) -> bytes: ...  # pragma: no cover

    async def exists(self, uri: str) -> bool: ...  # pragma: no cover

    async def public_url(self, uri: str) -> str | None: ...  # pragma: no cover


class DiskMediaStore:
    """Filesystem store. One file per blob at `<directory>/<sha256>.bin`.

    Directory is created on first write. Dedup is automatic via the
    content-hash filename.

    Pass `public_url=` to expose a URL for each blob — useful when a local
    HTTP server fronts the directory. Without it `public_url(...)` returns
    `None` (local filesystem paths are not addressable from a model).
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        public_url: PublicUrlResolver | None = None,
    ) -> None:
        self._root = Path(directory)
        self._public_url_resolver = public_url

    def _path_for(self, digest: str) -> Path:
        return self._root / f'{digest}.bin'

    async def put(self, data: bytes, *, media_type: str | None = None) -> str:
        return await anyio.to_thread.run_sync(self._sync_put, data)

    def _sync_put(self, data: bytes) -> str:
        uri = media_uri_for(data)
        digest = parse_media_uri(uri)
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._path_for(digest)
        if not path.exists():
            tmp = path.with_suffix('.bin.tmp')
            tmp.write_bytes(data)
            tmp.replace(path)
        return uri

    async def get(self, uri: str) -> bytes:
        digest = parse_media_uri(uri)
        return await anyio.to_thread.run_sync(self._sync_get, digest)

    def _sync_get(self, digest: str) -> bytes:
        path = self._path_for(digest)
        if not path.exists():
            raise FileNotFoundError(f'media not found: {digest}')
        return path.read_bytes()

    async def exists(self, uri: str) -> bool:
        digest = parse_media_uri(uri)
        return await anyio.to_thread.run_sync(self._sync_exists, digest)

    def _sync_exists(self, digest: str) -> bool:
        return self._path_for(digest).exists()

    async def public_url(self, uri: str) -> str | None:
        return await _resolve_public_url(self._public_url_resolver, uri)


class SqliteMediaStore:
    """SQLite store. One row per blob in a `media` table keyed by sha256 hex.

    Pass either a path to a SQLite file (created on demand) or an existing
    `sqlite3.Connection`. When a connection is passed the caller owns its
    lifecycle; when a path is passed each call opens a short-lived connection
    inside the worker thread (safe across event-loop threads).

    The table layout is:

    ```sql
    CREATE TABLE IF NOT EXISTS media (
        sha256 TEXT PRIMARY KEY,
        media_type TEXT,
        bytes BLOB NOT NULL,
        size_bytes INTEGER NOT NULL
    );
    ```

    `INSERT OR IGNORE` makes writes idempotent — the second `put` with the
    same content is a no-op, not an overwrite.
    """

    def __init__(
        self,
        *,
        database: str | Path | None = None,
        connection: sqlite3.Connection | None = None,
        table: str = 'media',
        public_url: PublicUrlResolver | None = None,
    ) -> None:
        if (database is None) == (connection is None):
            raise ValueError('provide exactly one of `database=` or `connection=`')
        self._database = Path(database) if database is not None else None
        self._connection = connection
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', table):
            raise ValueError(f'invalid table name: {table!r}')
        self._table = table
        self._schema_ready = False
        self._public_url_resolver = public_url

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        if self._schema_ready:
            return
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS {self._table} ('
            'sha256 TEXT PRIMARY KEY, '
            'media_type TEXT, '
            'bytes BLOB NOT NULL, '
            'size_bytes INTEGER NOT NULL)'
        )
        conn.commit()
        self._schema_ready = True

    def _open(self) -> sqlite3.Connection:
        if self._connection is not None:
            return self._connection
        assert self._database is not None
        self._database.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._database, isolation_level=None, check_same_thread=False)
        conn.execute('PRAGMA journal_mode=WAL').close()
        return conn

    def _maybe_close(self, conn: sqlite3.Connection) -> None:
        if self._connection is None:
            conn.close()

    async def put(self, data: bytes, *, media_type: str | None = None) -> str:
        return await anyio.to_thread.run_sync(self._sync_put, data, media_type)

    def _sync_put(self, data: bytes, media_type: str | None) -> str:
        uri = media_uri_for(data)
        digest = parse_media_uri(uri)
        conn = self._open()
        try:
            self._ensure_schema(conn)
            conn.execute(
                f'INSERT OR IGNORE INTO {self._table} (sha256, media_type, bytes, size_bytes) VALUES (?, ?, ?, ?)',
                (digest, media_type, data, len(data)),
            )
        finally:
            self._maybe_close(conn)
        return uri

    async def get(self, uri: str) -> bytes:
        digest = parse_media_uri(uri)
        return await anyio.to_thread.run_sync(self._sync_get, digest)

    def _sync_get(self, digest: str) -> bytes:
        conn = self._open()
        try:
            self._ensure_schema(conn)
            row = conn.execute(f'SELECT bytes FROM {self._table} WHERE sha256 = ?', (digest,)).fetchone()
        finally:
            self._maybe_close(conn)
        if row is None:
            raise FileNotFoundError(f'media not found: {digest}')
        return bytes(row[0])

    async def exists(self, uri: str) -> bool:
        digest = parse_media_uri(uri)
        return await anyio.to_thread.run_sync(self._sync_exists, digest)

    def _sync_exists(self, digest: str) -> bool:
        conn = self._open()
        try:
            self._ensure_schema(conn)
            row = conn.execute(f'SELECT 1 FROM {self._table} WHERE sha256 = ?', (digest,)).fetchone()
        finally:
            self._maybe_close(conn)
        return row is not None

    async def public_url(self, uri: str) -> str | None:
        return await _resolve_public_url(self._public_url_resolver, uri)
