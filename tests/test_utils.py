import pytest
import requests

from repurpose_verify.utils import CachedHTTP, download

BODY = b"0123456789" * 50


class FakeResponse:
    def __init__(self, data: bytes, status: int = 200, total: int | None = None):
        self.data, self.status_code = data, status
        self.headers = {"content-length": str(total if total is not None else len(data))}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self.data), chunk_size):
            yield self.data[i:i + chunk_size]

    def json(self):
        import json

        return json.loads(self.data)

    @property
    def text(self):
        return self.data.decode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_download_resumes(tmp_path, monkeypatch):
    dest = tmp_path / "f.bin"
    (tmp_path / "f.bin.part").write_bytes(BODY[:200])      # interrupted earlier
    seen = {}

    def fake_get(url, stream=False, timeout=None, headers=None):
        seen["range"] = headers.get("Range")
        return FakeResponse(BODY[200:], status=206)

    monkeypatch.setattr(requests, "get", fake_get)
    download(url="http://x/f.bin", dest=dest, chunk=64)
    assert seen["range"] == "bytes=200-"
    assert dest.read_bytes() == BODY
    assert not (tmp_path / "f.bin.part").exists()


def test_download_restarts_when_range_ignored(tmp_path, monkeypatch):
    dest = tmp_path / "g.bin"
    (tmp_path / "g.bin.part").write_bytes(b"stale data")
    monkeypatch.setattr(requests, "get",
                        lambda url, **kw: FakeResponse(BODY, status=200))
    download(url="http://x/g.bin", dest=dest, chunk=64)
    assert dest.read_bytes() == BODY


def test_download_resumes_after_truncated_stream(tmp_path, monkeypatch):
    """A stream that stops early is continued with a Range request, not accepted as complete."""
    calls = []

    def fake_get(url, stream=False, timeout=None, headers=None):
        calls.append(headers.get("Range"))
        if len(calls) == 1:
            return FakeResponse(BODY[:100], total=len(BODY))      # server cut us off
        return FakeResponse(BODY[100:], status=206)

    monkeypatch.setattr(requests, "get", fake_get)
    dest = tmp_path / "h.bin"
    download(url="http://x/h.bin", dest=dest, chunk=64)
    assert calls == [None, "bytes=100-"]
    assert dest.read_bytes() == BODY


def test_download_gives_up_after_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(requests, "get",
                        lambda url, **kw: FakeResponse(BODY[:100], total=len(BODY)))
    with pytest.raises(RuntimeError, match="Could not finish downloading"):
        download(url="http://x/i.bin", dest=tmp_path / "i.bin", chunk=64, attempts=2)


def test_download_skips_existing(tmp_path, monkeypatch):
    dest = tmp_path / "k.bin"
    dest.write_bytes(b"done")

    def boom(*a, **kw):
        raise AssertionError("should not fetch")

    monkeypatch.setattr(requests, "get", boom)
    assert download("http://x/k.bin", dest).read_bytes() == b"done"


def test_cached_http_caches_and_retries(tmp_path, monkeypatch):
    http = CachedHTTP(tmp_path / "cache", retries=3)
    calls = {"n": 0}

    def flaky(method, url, params=None, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse(b"{}", status=503)
        return FakeResponse(b'{"ok": 1}')

    monkeypatch.setattr(http._session, "request", flaky)
    assert http.get("http://x/api") == {"ok": 1}
    assert calls["n"] == 2
    assert http.get("http://x/api") == {"ok": 1}      # served from cache
    assert calls["n"] == 2


def test_cached_http_raises_on_client_error(tmp_path, monkeypatch):
    http = CachedHTTP(tmp_path / "cache")
    monkeypatch.setattr(http._session, "request",
                        lambda *a, **kw: FakeResponse(b"{}", status=404))
    with pytest.raises(requests.HTTPError):
        http.get("http://x/missing")
