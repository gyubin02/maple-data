from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from .utils import ApiMetrics, RateLimiter

BASE_URL = "https://open.api.nexon.com"


class ApiError(RuntimeError):
    def __init__(self, status_code: int, message: str, payload: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


class RateLimitError(ApiError):
    def __init__(self, status_code: int, message: str, payload: Any = None, retry_after: Optional[float] = None) -> None:
        super().__init__(status_code, message, payload)
        self.retry_after = retry_after


class ServerError(ApiError):
    pass


class DataPreparingError(ApiError):
    pass


class TransportError(RuntimeError):
    pass


@dataclass
class ApiClient:
    api_key: str
    concurrency: int = 8
    rps: float = 500.0
    timeout_seconds: float = 30.0
    max_attempts: int = 5

    def __post_init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(self.concurrency)
        self._rate_limiter = RateLimiter(self.rps)
        self.metrics = ApiMetrics()

    async def __aenter__(self) -> "ApiClient":
        headers = {"x-nxopen-api-key": self.api_key}
        self._client = httpx.AsyncClient(base_url=BASE_URL, headers=headers)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client:
            await self._client.aclose()

    async def get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        return await self._request_json("GET", path, params=params)

    async def _request_json(self, method: str, path: str, params: dict[str, Any]) -> dict[str, Any]:
        attempt = 0
        while True:
            attempt += 1
            try:
                await self._rate_limiter.acquire()
                async with self._semaphore:
                    assert self._client is not None
                    response = await self._client.request(
                        method,
                        path,
                        params=params,
                        timeout=self.timeout_seconds,
                    )
                self.metrics.total_requests += 1
                if 200 <= response.status_code < 300:
                    return response.json()

                payload = _safe_json(response)
                message = _extract_message(payload)
                if response.status_code == 400 and _is_data_preparing(payload):
                    self.metrics.data_preparing_hits += 1
                    raise DataPreparingError(response.status_code, message, payload)
                if response.status_code == 429:
                    self.metrics.rate_limit_hits += 1
                    retry_after = _retry_after_seconds(response)
                    raise RateLimitError(response.status_code, message, payload, retry_after=retry_after)
                if response.status_code >= 500:
                    self.metrics.server_errors += 1
                    raise ServerError(response.status_code, message, payload)

                self.metrics.other_errors += 1
                raise ApiError(response.status_code, message, payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                self.metrics.other_errors += 1
                error = TransportError(str(exc))
            except (RateLimitError, ServerError, DataPreparingError) as exc:
                error = exc
            except ApiError:
                raise

            if attempt >= self.max_attempts:
                raise error

            await asyncio.sleep(_compute_wait_seconds(error, attempt))


def _safe_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except json.JSONDecodeError:
        return {"message": response.text}


def _extract_message(payload: Any) -> str:
    if isinstance(payload, dict):
        if isinstance(payload.get("error"), dict):
            return payload["error"].get("message") or "API error"
        return payload.get("message") or "API error"
    return "API error"


def _extract_code(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        if isinstance(payload.get("error"), dict):
            return payload["error"].get("code") or payload["error"].get("error_code")
        return payload.get("code") or payload.get("error_code")
    return None


def _is_data_preparing(payload: Any) -> bool:
    code = _extract_code(payload)
    message = _extract_message(payload)
    if code and code.upper() == "OPENAPI00009":
        return True
    if message and "Data being prepared" in message:
        return True
    return False


def _retry_after_seconds(response: httpx.Response) -> Optional[float]:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _compute_wait_seconds(error: Exception, attempt: int) -> float:
    if isinstance(error, DataPreparingError):
        return random.uniform(30, 120)
    base = 1.0
    max_wait = 30.0
    wait = min(max_wait, base * (2 ** (attempt - 1)))
    jitter = random.uniform(0, 0.5)
    if isinstance(error, RateLimitError) and error.retry_after:
        return max(wait + jitter, error.retry_after)
    return wait + jitter
