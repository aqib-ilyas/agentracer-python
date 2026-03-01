import time
from . import _config, _current_feature_tag, track

try:
    import openai as _openai_lib

    def _wrap_openai_stream(stream, model, feature_tag, start):
        input_tokens = 0
        output_tokens = 0
        for chunk in stream:
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0
            yield chunk
        latency = (time.time() - start) * 1000
        try:
            track(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency,
                feature_tag=feature_tag,
                provider="openai"
            )
        except Exception:
            pass

    async def _wrap_openai_stream_async(stream, model, feature_tag, start):
        input_tokens = 0
        output_tokens = 0
        async for chunk in stream:
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens or 0
                output_tokens = chunk.usage.completion_tokens or 0
            yield chunk
        latency = (time.time() - start) * 1000
        try:
            track(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency,
                feature_tag=feature_tag,
                provider="openai"
            )
        except Exception:
            pass

    class _TrackedCompletions:
        def __init__(self, original):
            self._original = original

        def create(self, **kwargs):
            feature_tag = kwargs.pop("feature_tag", None) or _current_feature_tag.get() or "unknown"

            if kwargs.get("stream"):
                kwargs.setdefault("stream_options", {})
                kwargs["stream_options"]["include_usage"] = True
                start = time.time()
                stream = self._original.create(**kwargs)
                return _wrap_openai_stream(stream, kwargs.get("model", "unknown"), feature_tag, start)

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
                    provider="openai",
                    success=False,
                    error_type=type(exc).__name__,
                )
                raise
            latency = (time.time() - start) * 1000

            try:
                usage = response.usage
                cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
                track(
                    model=kwargs.get("model", "unknown"),
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    cached_tokens=cached,
                    latency_ms=latency,
                    feature_tag=feature_tag,
                    provider="openai"
                )
            except Exception:
                pass
            return response

    class _TrackedAsyncCompletions:
        def __init__(self, original):
            self._original = original

        async def create(self, **kwargs):
            feature_tag = kwargs.pop("feature_tag", None) or _current_feature_tag.get() or "unknown"

            if kwargs.get("stream"):
                kwargs.setdefault("stream_options", {})
                kwargs["stream_options"]["include_usage"] = True
                start = time.time()
                stream = await self._original.create(**kwargs)
                return _wrap_openai_stream_async(stream, kwargs.get("model", "unknown"), feature_tag, start)

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
                    provider="openai",
                    success=False,
                    error_type=type(exc).__name__,
                )
                raise
            latency = (time.time() - start) * 1000

            try:
                usage = response.usage
                cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
                track(
                    model=kwargs.get("model", "unknown"),
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    cached_tokens=cached,
                    latency_ms=latency,
                    feature_tag=feature_tag,
                    provider="openai"
                )
            except Exception:
                pass
            return response

    class _TrackedChat:
        def __init__(self, original):
            self._original = original

        @property
        def completions(self):
            return _TrackedCompletions(self._original.completions)

    class _TrackedAsyncChat:
        def __init__(self, original):
            self._original = original

        @property
        def completions(self):
            return _TrackedAsyncCompletions(self._original.completions)

    class _TrackedOpenAI:
        def __init__(self, **kwargs):
            self._kwargs = kwargs
            self._client = None

        def _get_client(self):
            if not self._client:
                self._client = _openai_lib.OpenAI(**self._kwargs)
            return self._client

        @property
        def chat(self):
            return _TrackedChat(self._get_client().chat)

        def __getattr__(self, name):
            return getattr(self._get_client(), name)

    class _TrackedAsyncOpenAI:
        def __init__(self, **kwargs):
            self._kwargs = kwargs
            self._client = None

        def _get_client(self):
            if not self._client:
                self._client = _openai_lib.AsyncOpenAI(**self._kwargs)
            return self._client

        @property
        def chat(self):
            return _TrackedAsyncChat(self._get_client().chat)

        def __getattr__(self, name):
            return getattr(self._get_client(), name)

    TrackedOpenAI = _TrackedOpenAI
    TrackedAsyncOpenAI = _TrackedAsyncOpenAI
    openai = _TrackedOpenAI()

except ImportError:
    raise ImportError("openai package not installed. Run: pip install openai")
