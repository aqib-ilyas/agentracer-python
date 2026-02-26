import time
from . import _config, _current_feature_tag, track

try:
    import google.generativeai as _genai_lib

    def _wrap_gemini_stream(stream, model_name, feature_tag, start):
        input_tokens = 0
        output_tokens = 0
        for chunk in stream:
            usage = getattr(chunk, "usage_metadata", None)
            if usage:
                input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                output_tokens = getattr(usage, "candidates_token_count", 0) or 0
            yield chunk
        latency = (time.time() - start) * 1000
        try:
            track(
                model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency,
                feature_tag=feature_tag,
                provider="gemini"
            )
        except Exception:
            pass

    class _TrackedGenerativeModel:
        def __init__(self, model_name: str, **kwargs):
            self._model_name = model_name
            self._model = _genai_lib.GenerativeModel(model_name, **kwargs)

        def generate_content(self, *args, feature_tag: str = None, **kwargs):
            tag = feature_tag or _current_feature_tag.get() or "unknown"

            if kwargs.get("stream"):
                start = time.time()
                stream = self._model.generate_content(*args, **kwargs)
                return _wrap_gemini_stream(stream, self._model_name, tag, start)

            start = time.time()
            response = self._model.generate_content(*args, **kwargs)
            latency = (time.time() - start) * 1000

            try:
                usage = response.usage_metadata
                track(
                    model=self._model_name,
                    input_tokens=usage.prompt_token_count,
                    output_tokens=usage.candidates_token_count,
                    latency_ms=latency,
                    feature_tag=tag,
                    provider="gemini"
                )
            except Exception:
                pass
            return response

        def __getattr__(self, name):
            return getattr(self._model, name)

    class _TrackedGemini:
        def GenerativeModel(self, model_name: str, **kwargs):
            return _TrackedGenerativeModel(model_name, **kwargs)

        def __getattr__(self, name):
            return getattr(_genai_lib, name)

    gemini = _TrackedGemini()

except ImportError:
    raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
