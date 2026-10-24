"""Shared helpers: cached HTTP, JSONL I/O, downloads, robust JSON extraction."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import requests

log = logging.getLogger("repurpose_verify")

USER_AGENT = "repurpose-verify/0.1 (research benchmark)"


class CachedHTTP:
    """requests wrapper with an on-disk JSON cache, retries and a per-host rate limit.

    Every external lookup goes through the cache so that a benchmark run is reproducible
    offline once the cache is populated, and so that the exact evidence seen by the agent
    can be archived with the paper.
    """

    def __init__(self, cache_dir: Path, min_interval: float = 0.0, timeout: float = 60.0,
                 retries: int = 4):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.timeout = timeout
        self.retries = retries
        self._session = requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT
        self._lock = threading.Lock()
        self._last = 0.0

    def _key(self, method: str, url: str, params: dict | None, body: Any) -> Path:
        blob = json.dumps([method, url, params or {}, body], sort_keys=True, default=str)
        digest = hashlib.sha1(blob.encode()).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.json"

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def request(self, method: str, url: str, params: dict | None = None, json_body: Any = None,
                as_json: bool = True, use_cache: bool = True) -> Any:
        path = self._key(method, url, params, json_body)
        if use_cache and path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload["data"]
        last_err: Exception | None = None
        for attempt in range(self.retries):
            self._throttle()
            try:
                resp = self._session.request(method, url, params=params, json=json_body,
                                             timeout=self.timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                data = resp.json() if as_json else resp.text
                break
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as err:
                status = getattr(getattr(err, "response", None), "status_code", None)
                if status is not None and status < 500 and status != 429:
                    raise
                last_err = err
                time.sleep(min(2 ** attempt, 30))
        else:
            raise RuntimeError(f"Request failed after {self.retries} attempts: {url}") from last_err
        if use_cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"url": url, "params": params, "data": data}),
                            encoding="utf-8")
        return data

    def get(self, url: str, params: dict | None = None, **kw: Any) -> Any:
        return self.request("GET", url, params=params, **kw)

    def post(self, url: str, json_body: Any, **kw: Any) -> Any:
        return self.request("POST", url, json_body=json_body, **kw)


def download(url: str, dest: Path, chunk: int = 1 << 20, attempts: int = 10) -> Path:
    """Stream a large file to disk, resuming a partial ``.part`` with a Range request.

    Downloads here are hundreds of megabytes over links that may be slow or shared, so an
    interrupted transfer continues instead of starting over.
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        log.info("Already downloaded: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    from tqdm import tqdm

    done = tmp.stat().st_size if tmp.exists() else 0
    total: int | None = None
    bar = tqdm(total=None, initial=done, unit="B", unit_scale=True, desc=dest.name)
    try:
        for attempt in range(attempts):
            headers = {"User-Agent": USER_AGENT}
            if done:
                headers["Range"] = f"bytes={done}-"
            try:
                with requests.get(url, stream=True, timeout=120, headers=headers) as r:
                    if done and r.status_code == 416:  # already complete
                        break
                    r.raise_for_status()
                    if done and r.status_code != 206:  # server ignored Range: start over
                        log.warning("Range not honoured for %s; restarting", dest.name)
                        done, mode = 0, "wb"
                        bar.reset()
                    else:
                        mode = "ab" if done else "wb"
                    length = int(r.headers.get("content-length", 0)) or None
                    if length:
                        total = done + length
                        bar.total = total
                        bar.refresh()
                    with open(tmp, mode) as f:
                        for part in r.iter_content(chunk_size=chunk):
                            f.write(part)
                            done += len(part)
                            bar.update(len(part))
                if total is None or done >= total:
                    break
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as err:
                done = tmp.stat().st_size if tmp.exists() else 0
                if attempt == attempts - 1:
                    raise
                wait = min(2 ** attempt, 30)
                log.warning("%s: %s - resuming from %d bytes in %ds", dest.name, err, done, wait)
                time.sleep(wait)
        else:
            raise RuntimeError(f"Could not finish downloading {url}")
    finally:
        bar.close()
    if total is not None and tmp.stat().st_size != total:
        raise RuntimeError(f"Incomplete download: {tmp} ({tmp.stat().st_size} of {total} bytes)")
    tmp.replace(dest)
    return dest


def read_jsonl(path: Path | str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path | str, rows: Iterable[dict]) -> None:
    """Rewrite a JSONL file, via a temporary file so a crash cannot truncate the original.

    Callers pass this run records that took hours to produce; a half-written file would be
    worse than no write at all.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict, lock: threading.Lock | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    if lock:
        with lock, open(path, "a", encoding="utf-8") as f:
            f.write(line)
    else:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in an LLM response (fenced or bare)."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            start = cand.find(opener)
            while start != -1:
                depth, in_str, esc = 0, False, False
                for i in range(start, len(cand)):
                    ch = cand[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                    elif ch == '"':
                        in_str = True
                    elif ch == opener:
                        depth += 1
                    elif ch == closer:
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(cand[start:i + 1])
                            except json.JSONDecodeError:
                                break
                start = cand.find(opener, start + 1)
    raise ValueError("No JSON found in model output")
