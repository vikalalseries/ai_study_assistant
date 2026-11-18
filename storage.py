"""Vercel Blob storage for the notes library (the REST API, so no extra SDK is needed).

Vercel's filesystem is read-only except /tmp, and /tmp is wiped between instances, so the uploaded notes live in
Blob (the source of truth) and are copied into a local cache folder that the indexer reads.
Needs BLOB_READ_WRITE_TOKEN in .env / the Vercel project. Without it, enabled() is False and notes stay local-only.
"""
import os
import time
from urllib.parse import quote

import httpx

API = "https://blob.vercel-storage.com"
PREFIX = "library/"
TIMEOUT = 30


def _token() -> str:
    return os.getenv("BLOB_READ_WRITE_TOKEN", "")


def enabled() -> bool:
    return bool(_token())


def _headers(**extra: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {_token()}", "x-api-version": "7", **extra}


def _check(resp: httpx.Response) -> httpx.Response:
    if resp.status_code >= 400:
        raise RuntimeError(f"Vercel Blob error {resp.status_code}: {resp.text[:200]}")
    return resp


def _request(method: str, url: str, **kw) -> httpx.Response:
    """httpx request that retries transient network errors (connection resets happen on Blob now and then)."""
    for attempt in range(3):
        try:
            return httpx.request(method, url, timeout=TIMEOUT, **kw)
        except httpx.TransportError:
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))


def list_files() -> dict[str, dict]:
    """{filename: {"url", "size", "uploaded"}} for every note stored in Blob."""
    out, cursor = {}, None
    while True:
        params = {"prefix": PREFIX, "limit": "1000"}
        if cursor:
            params["cursor"] = cursor
        data = _check(_request("GET", API, params=params, headers=_headers())).json()
        for b in data.get("blobs", []):
            name = b["pathname"][len(PREFIX):]
            if name:
                out[name] = {"url": b.get("downloadUrl") or b["url"], "size": b.get("size", 0),
                             "uploaded": str(b.get("uploadedAt", ""))}
        if not data.get("hasMore"):
            return out
        cursor = data.get("cursor")


def put(name: str, data: bytes) -> None:
    _check(_request(
        "PUT", f"{API}/{quote(PREFIX + name)}",
        content=data,
        headers=_headers(**{
            "x-add-random-suffix": "0",       # keep the exact filename so re-saving overwrites
            "x-allow-overwrite": "1",
            "x-cache-control-max-age": "60",  # don't let the CDN serve an old version for a month
            "x-content-type": "text/markdown; charset=utf-8" if name.endswith(".md") else "text/plain; charset=utf-8",
        }),
    ))


def delete(name: str) -> None:
    info = list_files().get(name)
    if info:
        _check(_request("POST", f"{API}/delete", json={"urls": [info["url"]]}, headers=_headers()))


def download(info: dict) -> bytes:
    # The uploaded timestamp in the query string busts any CDN copy of an earlier version.
    kw = {"params": {"v": info["uploaded"]}, "follow_redirects": True}
    for attempt in range(4):
        resp = _request("GET", info["url"], **kw)  # public stores: no auth (sending it there causes a 404)
        if resp.status_code in (401, 403):
            resp = _request("GET", info["url"], headers=_headers(), **kw)  # private stores need the token
        if resp.status_code != 404:
            break
        time.sleep(0.75 * (attempt + 1))  # a just-uploaded blob can take a moment to reach the CDN
    return _check(resp).content
