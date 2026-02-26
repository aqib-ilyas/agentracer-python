import asyncio
import functools
import time
from contextvars import ContextVar
from typing import Optional

import httpx

__version__ = "0.1.0"
__all__ = ["init", "observe", "feature_context", "track"]

_config = {}
_current_feature_tag: ContextVar[Optional[str]] = ContextVar('feature_tag', default=None)


def init(
    tracker_api_key: str,
    project_id: str,
    environment: str = "production",
    host: str = "https://api.agentracer.dev",
    debug: bool = False,
    enabled: bool = True
):
    _config.update({
        "tracker_api_key": tracker_api_key,
        "project_id": project_id,
        "environment": environment,
        "host": host,
        "debug": debug,
        "enabled": enabled
    })


def observe(func=None, *, feature_tag: str = "unknown"):
    def decorator(f):
        @functools.wraps(f)
        async def async_wrapper(*args, **kwargs):
            token = _current_feature_tag.set(feature_tag)
            try:
                return await f(*args, **kwargs)
            finally:
                _current_feature_tag.reset(token)

        @functools.wraps(f)
        def sync_wrapper(*args, **kwargs):
            token = _current_feature_tag.set(feature_tag)
            try:
                return f(*args, **kwargs)
            finally:
                _current_feature_tag.reset(token)

        return async_wrapper if asyncio.iscoroutinefunction(f) else sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


class feature_context:
    def __init__(self, tag: str):
        self.tag = tag
        self.token = None

    def __enter__(self):
        self.token = _current_feature_tag.set(self.tag)
        return self

    def __exit__(self, *args):
        _current_feature_tag.reset(self.token)


async def _send_telemetry(payload: dict):
    if not _config.get("enabled", True):
        return
    try:
        if _config.get("debug"):
            print(f"[agentracer] {payload}")
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{_config['host']}/api/ingest",
                json=payload,
                headers={"x-api-key": _config["tracker_api_key"]},
                timeout=2.0
            )
    except Exception:
        pass


def _send_telemetry_sync(payload: dict):
    if not _config.get("enabled", True):
        return
    try:
        if _config.get("debug"):
            print(f"[agentracer] {payload}")
        with httpx.Client() as client:
            client.post(
                f"{_config['host']}/api/ingest",
                json=payload,
                headers={"x-api-key": _config["tracker_api_key"]},
                timeout=2.0
            )
    except Exception:
        pass


def track(
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    feature_tag: str = "unknown",
    environment: Optional[str] = None,
    provider: str = "custom"
):
    if not _config.get("enabled", True):
        return

    payload = {
        "project_id": _config["project_id"],
        "provider": provider,
        "model": model,
        "feature_tag": feature_tag or _current_feature_tag.get() or "unknown",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "environment": environment or _config.get("environment", "production")
    }

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_send_telemetry(payload))
    except RuntimeError:
        import threading
        threading.Thread(target=_send_telemetry_sync, args=(payload,), daemon=True).start()
