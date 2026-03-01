import asyncio
import functools
import time
import uuid as _uuid_mod
from contextvars import ContextVar
from typing import Optional

import httpx

__version__ = "0.1.0"
__all__ = ["init", "observe", "feature_context", "track", "AgentRun", "__version__"]

_config = {}
_current_feature_tag: ContextVar[Optional[str]] = ContextVar('feature_tag', default=None)
_current_run: ContextVar[Optional["AgentRun"]] = ContextVar('current_run', default=None)


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
    feature_tag: Optional[str] = None,
    environment: Optional[str] = None,
    provider: str = "custom",
    cached_tokens: int = 0,
    success: bool = True,
    error_type: Optional[str] = None,
    end_user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    step_index: Optional[int] = None,
):
    if not _config.get("enabled", True):
        return

    # Auto-detect active AgentRun
    active_run = _current_run.get()
    if active_run is not None and run_id is None:
        run_id = active_run.run_id
        step_index = active_run._next_step()
        # Also send step to runs API
        _send_run_step(active_run, step_index, model, provider, input_tokens,
                       output_tokens, cached_tokens, latency_ms, success,
                       error_type, "llm_call", None, False)

    resolved_tag = feature_tag or _current_feature_tag.get() or "unknown"

    payload = {
        "project_id": _config["project_id"],
        "provider": provider,
        "model": model,
        "feature_tag": resolved_tag,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "latency_ms": latency_ms,
        "success": success,
        "environment": environment or _config.get("environment", "production"),
    }

    if error_type is not None:
        payload["error_type"] = error_type
    if end_user_id is not None:
        payload["end_user_id"] = end_user_id
    if run_id is not None:
        payload["run_id"] = run_id
    if step_index is not None:
        payload["step_index"] = step_index

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_send_telemetry(payload))
    except RuntimeError:
        import threading
        threading.Thread(target=_send_telemetry_sync, args=(payload,), daemon=True).start()


def _send_run_step(run, step_index, model, provider, input_tokens, output_tokens,
                   cached_tokens, latency_ms, success, error_type, step_type, tool_name, is_retry):
    """Fire-and-forget step recording to runs API."""
    payload = {
        "project_id": _config["project_id"],
        "run_id": run.run_id,
        "step_index": step_index,
        "step_type": step_type or "llm_call",
        "model": model,
        "provider": provider,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "cost_usd": 0,  # cost calculated server-side via ingest
        "latency_ms": latency_ms,
        "success": success,
        "error_type": error_type,
        "tool_name": tool_name,
        "is_retry": is_retry or False,
    }
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_send_run_api(payload, "/api/runs/step"))
    except RuntimeError:
        import threading
        threading.Thread(target=_send_run_api_sync, args=(payload, "/api/runs/step"), daemon=True).start()


async def _send_run_api(payload: dict, path: str):
    if not _config.get("enabled", True):
        return
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{_config['host']}{path}",
                json=payload,
                headers={"x-api-key": _config["tracker_api_key"]},
                timeout=2.0,
            )
    except Exception:
        pass


def _send_run_api_sync(payload: dict, path: str):
    if not _config.get("enabled", True):
        return
    try:
        with httpx.Client() as client:
            client.post(
                f"{_config['host']}{path}",
                json=payload,
                headers={"x-api-key": _config["tracker_api_key"]},
                timeout=2.0,
            )
    except Exception:
        pass


class AgentRun:
    """Context manager for tracking multi-step agent runs.

    Usage:
        with AgentRun(run_name="summarize", feature_tag="pdf-bot") as run:
            response = openai.chat.completions.create(...)
            # Steps are auto-tracked
    """

    def __init__(
        self,
        run_name: Optional[str] = None,
        feature_tag: str = "unknown",
        end_user_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        self.run_id = run_id or str(_uuid_mod.uuid4())
        self.run_name = run_name
        self.feature_tag = feature_tag
        self.end_user_id = end_user_id
        self._step_counter = 0
        self._token = None
        self._feature_token = None

    def _next_step(self) -> int:
        self._step_counter += 1
        return self._step_counter

    def __enter__(self):
        self._token = _current_run.set(self)
        if self.feature_tag:
            self._feature_token = _current_feature_tag.set(self.feature_tag)

        # Fire start request
        payload = {
            "project_id": _config["project_id"],
            "run_id": self.run_id,
            "run_name": self.run_name,
            "feature_tag": self.feature_tag,
            "end_user_id": self.end_user_id,
        }
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_send_run_api(payload, "/api/runs/start"))
        except RuntimeError:
            import threading
            threading.Thread(target=_send_run_api_sync, args=(payload, "/api/runs/start"), daemon=True).start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "failed" if exc_type else "completed"
        error = type(exc_val).__name__ if exc_val else None

        payload = {
            "project_id": _config["project_id"],
            "run_id": self.run_id,
            "status": status,
            "error_type": error,
        }
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_send_run_api(payload, "/api/runs/end"))
        except RuntimeError:
            import threading
            threading.Thread(target=_send_run_api_sync, args=(payload, "/api/runs/end"), daemon=True).start()

        if self._feature_token:
            _current_feature_tag.reset(self._feature_token)
        _current_run.reset(self._token)

        return False  # don't suppress exceptions

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)
