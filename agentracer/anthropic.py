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
            response = self._original.create(**kwargs)
            latency = (time.time() - start) * 1000

            try:
                track(
                    model=kwargs.get("model", "unknown"),
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    latency_ms=latency,
                    feature_tag=feature_tag,
                    provider="anthropic"
                )
            except Exception:
                pass
            return response

    class _TrackedAnthropic:
        def __init__(self):
            self._client = None

        def _get_client(self):
            if not self._client:
                self._client = _anthropic_lib.Anthropic()
            return self._client

        @property
        def messages(self):
            return _TrackedMessages(self._get_client().messages)

        def __getattr__(self, name):
            return getattr(self._get_client(), name)

    anthropic = _TrackedAnthropic()

except ImportError:
    raise ImportError("anthropic package not installed. Run: pip install anthropic")
