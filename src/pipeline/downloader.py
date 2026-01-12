from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx

from .utils import DownloadResult, RateLimiter, guess_extension, random_wait, utc_now_iso


@dataclass
class DownloadRecord:
    url: str
    sha256: Optional[str]
    local_path: Optional[str]
    content_type: Optional[str]
    byte_size: Optional[int]
    fetched_at: Optional[str]
    error: Optional[str]


async def download_icons(
    urls_by_category: dict[str, str],
    output_root: Path,
    icons_root: Path,
    existing_assets: dict[str, dict[str, Any]],
    rps: float,
    concurrency: int,
    max_attempts: int = 5,
) -> tuple[list[DownloadRecord], DownloadResult]:
    limiter = RateLimiter(rps)
    semaphore = asyncio.Semaphore(concurrency)
    results: list[DownloadRecord] = []
    counters = DownloadResult()

    async with httpx.AsyncClient() as client:
        tasks = []
        for url, category in urls_by_category.items():
            if not url:
                continue
            existing = existing_assets.get(url)
            if existing and existing.get("sha256"):
                counters.skipped += 1
                continue
            tasks.append(
                asyncio.create_task(
                    _download_one(
                        client,
                        limiter,
                        semaphore,
                        url,
                        category,
                        output_root,
                        icons_root,
                        max_attempts,
                    )
                )
            )
        if tasks:
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            for item in completed:
                if isinstance(item, Exception):
                    counters.failed += 1
                    results.append(
                        DownloadRecord(
                            url="unknown",
                            sha256=None,
                            local_path=None,
                            content_type=None,
                            byte_size=None,
                            fetched_at=utc_now_iso(),
                            error=str(item),
                        )
                    )
                    continue
                record, status = item
                results.append(record)
                if status == "downloaded":
                    counters.downloaded += 1
                elif status == "failed":
                    counters.failed += 1
    return results, counters


async def _download_one(
    client: httpx.AsyncClient,
    limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
    url: str,
    category: str,
    output_root: Path,
    icons_root: Path,
    max_attempts: int,
) -> tuple[DownloadRecord, str]:
    attempt = 0
    while True:
        attempt += 1
        try:
            await limiter.acquire()
            async with semaphore:
                response = await client.get(url, timeout=30)
            if 200 <= response.status_code < 300:
                content = response.content
                sha256 = hashlib.sha256(content).hexdigest()
                extension = guess_extension(response.headers.get("content-type"), url)
                target_dir = icons_root / category
                target_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{sha256}{extension}"
                path = target_dir / filename
                if not path.exists():
                    path.write_bytes(content)
                local_path = str(path.relative_to(output_root))
                return (
                    DownloadRecord(
                        url=url,
                        sha256=sha256,
                        local_path=local_path,
                        content_type=response.headers.get("content-type"),
                        byte_size=len(content),
                        fetched_at=utc_now_iso(),
                        error=None,
                    ),
                    "downloaded",
                )
            if response.status_code == 429 or response.status_code >= 500:
                raise RuntimeError(f"HTTP {response.status_code}")
            return (
                DownloadRecord(
                    url=url,
                    sha256=None,
                    local_path=None,
                    content_type=response.headers.get("content-type"),
                    byte_size=None,
                    fetched_at=utc_now_iso(),
                    error=f"HTTP {response.status_code}",
                ),
                "failed",
            )
        except (httpx.TimeoutException, httpx.TransportError, RuntimeError) as exc:
            if attempt >= max_attempts:
                return (
                    DownloadRecord(
                        url=url,
                        sha256=None,
                        local_path=None,
                        content_type=None,
                        byte_size=None,
                        fetched_at=utc_now_iso(),
                        error=str(exc),
                    ),
                    "failed",
                )
            await asyncio.sleep(_download_backoff(attempt))


def _download_backoff(attempt: int) -> float:
    base = 1.0
    max_wait = 20.0
    wait = min(max_wait, base * (2 ** (attempt - 1)))
    return wait + random_wait(0, 0.5)
