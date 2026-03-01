import time
from . import _config, _current_feature_tag, track

try:
    import anthropic as _anthropic_lib

    def _wrap_anthropic_stream(stream, model, feature_tag, start):
        input_tokens = 0
        output_tokens = 0
        for event in stream:
            event_type = getattr(event, "type", None)
            if event_type == "message_start":
                usage = getattr(getattr(event, "message", None), "usage", None)
                if usage:
                    input_tokens = getattr(usage, "input_tokens", 0) or 0
            if event_type == "message_delta":
                usage = getattr(event, "usage", None)
                if usage:
                    output_tokens = getattr(usage, "output_tokens", 0) or 0
            yield event
        latency = (time.time() - start) * 1000
        try:
            track(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency,
                feature_tag=feature_tag,
                provider="anthropic"
            )
        except Exception:
            pass

    async def _wrap_anthropic_stream_async(stream, model, feature_tag, start):
        input_tokens = 0
        output_tokens = 0
        async for event in stream:
            event_type = getattr(event, "type", None)
            if event_type == "message_start":
                usage = getattr(getattr(event, "message", None), "usage", None)
                if usage:
                    input_tokens = getattr(usage, "input_tokens", 0) or 0
            if event_type == "message_delta":
                usage = getattr(event, "usage", None)
                if usage:
                    output_tokens = getattr(usage, "output_tokens", 0) or 0
            yield event
        latency = (time.time() - start) * 1000
        try:
            track(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency,
                feature_tag=feature_tag,
                provider="anthropic"
            )
        except Exception:
            pass

    class _TrackedMessages:
        def __init__(self, original):
            self._original = original

        def create(self, **kwargs):
            feature_tag = kwargs.pop("feature_tag", None) or _current_feature_tag.get() or "unknown"

            if kwargs.get("stream"):
                start = time.time()
                stream = self._original.create(**kwargs)
                return _wrap_anthropic_stream(stream, kwargs.get("model", "unknown"), feature_tag, start)

            start = time.time()
            try:
                response = self._original.create(**kwargs)
            except Exception as exc:
                latency = (time.time() - start) * 1000
                track(
                    model=kwargs.get("model", "unknown"),
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=latency,
                    feature_tag=feature_tag,
                    provider="anthropic",
                    success=False,
                    error_type=type(exc).__name__,
                )
                raise
            latency = (time.time() - start) * 1000

            try:
                usage = response.usage
                cached = getattr(usage, "cache_read_input_tokens", 0) or 0
                track(
                    model=kwargs.get("model", "unknown"),
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cached_tokens=cached,
                    latency_ms=latency,
                    feature_tag=feature_tag,
                    provider="anthropic"
                )
            except Exception:
                pass
            return response

    class _TrackedAsyncMessages:
        def __init__(self, original):
            self._original = original

        async def create(self, **kwargs):
            feature_tag = kwargs.pop("feature_tag", None) or _current_feature_tag.get() or "unknown"

            if kwargs.get("stream"):
                start = time.time()
                stream = await self._original.create(**kwargs)
                return _wrap_anthropic_stream_async(stream, kwargs.get("model", "unknown"), feature_tag, start)

            start = time.time()
            try:
                response = await self._original.create(**kwargs)
            except Exception as exc:
                latency = (time.time() - start) * 1000
                track(
                    model=kwargs.get("model", "unknown"),
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=latency,
                    feature_tag=feature_tag,
                    provider="anthropic",
                    success=False,
                    error_type=type(exc).__name__,
                )
                raise
            latency = (time.time() - start) * 1000

            try:
                usage = response.usage
                cached = getattr(usage, "cache_read_input_tokens", 0) or 0
                track(
                    model=kwargs.get("model", "unknown"),
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cached_tokens=cached,
                    latency_ms=latency,
                    feature_tag=feature_tag,
                    provider="anthropic"
                )
            except Exception:
                pass
            return response

    class _TrackedAnthropic:
        def __init__(self, **kwargs):
            self._kwargs = kwargs
            self._client = None

        def _get_client(self):
            if not self._client:
                self._client = _anthropic_lib.Anthropic(**self._kwargs)
            return self._client

        @property
        def messages(self):
            return _TrackedMessages(self._get_client().messages)

        def __getattr__(self, name):
            return getattr(self._get_client(), name)

    class _TrackedAsyncAnthropic:
        def __init__(self, **kwargs):
            self._kwargs = kwargs
            self._client = None

        def _get_client(self):
            if not self._client:
                self._client = _anthropic_lib.AsyncAnthropic(**self._kwargs)
            return self._client

        @property
        def messages(self):
            return _TrackedAsyncMessages(self._get_client().messages)

        def __getattr__(self, name):
            return getattr(self._get_client(), name)

    TrackedAnthropic = _TrackedAnthropic
    TrackedAsyncAnthropic = _TrackedAsyncAnthropic
    anthropic = _TrackedAnthropic()

except ImportError:
    raise ImportError("anthropic package not installed. Run: pip install anthropic")
