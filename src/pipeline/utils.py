from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote
from zoneinfo import ZoneInfo


def load_dotenv_if_available() -> bool:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False

    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        return True
    return False


def kst_yesterday_date() -> str:
    tz = ZoneInfo("Asia/Seoul")
    now = datetime.now(tz)
    yesterday = (now - timedelta(days=1)).date()
    return yesterday.isoformat()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(value: str) -> str:
    if not value:
        return "unknown"
    return quote(value, safe="-_.")


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def compute_run_id(target_date: str, params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{target_date}-{digest}"


def to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def guess_extension(content_type: Optional[str], url: str) -> str:
    if content_type:
        ext = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if ext:
            return ext
    suffix = Path(url).suffix
    if suffix:
        return suffix
    return ".bin"


def random_wait(min_seconds: float, max_seconds: float) -> float:
    return random.uniform(min_seconds, max_seconds)


@dataclass
class RateLimiter:
    rps: float

    def __post_init__(self) -> None:
        self._lock = None
        self._next_time: Optional[float] = None

    async def acquire(self) -> None:
        if self.rps <= 0:
            return
        if self._lock is None:
            import asyncio

            self._lock = asyncio.Lock()
        async with self._lock:
            now = time.monotonic()
            min_interval = 1 / self.rps
            if self._next_time is None:
                self._next_time = now
            if now < self._next_time:
                sleep_for = self._next_time - now
                if sleep_for > 0:
                    import asyncio

                    await asyncio.sleep(sleep_for)
                now = time.monotonic()
            self._next_time = max(now, self._next_time) + min_interval


@dataclass
class DownloadResult:
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0


@dataclass
class ApiMetrics:
    total_requests: int = 0
    rate_limit_hits: int = 0
    server_errors: int = 0
    data_preparing_hits: int = 0
    other_errors: int = 0


@dataclass
class PipelineReport:
    run_id: str
    target_date: str
    start_rank: int
    end_rank: int
    ranking_count: int
    ocid_count: int
    equipment_items_count: int
    cash_items_count: int
    icons_downloaded: int
    icons_skipped: int
    icons_failed: int
    rate_limit_hits: int
    server_errors: int
    data_preparing_hits: int
    elapsed_seconds: float

    def to_markdown(self) -> str:
        return "\n".join(
            [
                f"Run ID: {self.run_id}",
                f"Target date (KST): {self.target_date}",
                f"Rank range: {self.start_rank}-{self.end_rank}",
                f"Ranking entries: {self.ranking_count}",
                f"OCIDs resolved: {self.ocid_count}",
                f"Equipment shape items: {self.equipment_items_count}",
                f"Cash items: {self.cash_items_count}",
                f"Icons downloaded: {self.icons_downloaded}",
                f"Icons skipped: {self.icons_skipped}",
                f"Icons failed: {self.icons_failed}",
                f"429 retries: {self.rate_limit_hits}",
                f"5xx retries: {self.server_errors}",
                f"Data preparing retries: {self.data_preparing_hits}",
                f"Elapsed seconds: {self.elapsed_seconds:.2f}",
            ]
        )


def get_env_or_none(key: str) -> Optional[str]:
    value = os.getenv(key)
    return value if value else None
