import sys
from unittest.mock import MagicMock, patch
import agentracer

# Set up mock openai module before importing agentracer.openai
_mock_openai_module = MagicMock()
sys.modules["openai"] = _mock_openai_module

from agentracer.openai import openai, TrackedOpenAI  # noqa: E402

# The track function reference that agentracer.openai captured at import time
TRACK_PATCH_TARGET = "agentracer.openai.track"


class TestOpenAINonStreaming:
    def setup_method(self):
        agentracer.init(tracker_api_key="k", project_id="proj-1", enabled=True)
        _mock_openai_module.OpenAI.return_value.chat.completions.create.reset_mock()

    def test_sends_telemetry_with_correct_tokens(self):
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        _mock_openai_module.OpenAI.return_value.chat.completions.create.return_value = mock_response

        with patch(TRACK_PATCH_TARGET) as mock_track:
            response = openai.chat.completions.create(model="gpt-4", messages=[])

        assert response is mock_response
        mock_track.assert_called_once()
        kwargs = mock_track.call_args[1]
        assert kwargs["model"] == "gpt-4"
        assert kwargs["input_tokens"] == 10
        assert kwargs["output_tokens"] == 20
        assert kwargs["provider"] == "openai"

    def test_uses_feature_tag_from_context(self):
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 1
        mock_response.usage.completion_tokens = 1
        _mock_openai_module.OpenAI.return_value.chat.completions.create.return_value = mock_response

        with patch(TRACK_PATCH_TARGET) as mock_track:
            with agentracer.feature_context("search"):
                openai.chat.completions.create(model="gpt-4", messages=[])

        kwargs = mock_track.call_args[1]
        assert kwargs["feature_tag"] == "search"

    def test_explicit_feature_tag_overrides_context(self):
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 1
        mock_response.usage.completion_tokens = 1
        _mock_openai_module.OpenAI.return_value.chat.completions.create.return_value = mock_response

        with patch(TRACK_PATCH_TARGET) as mock_track:
            with agentracer.feature_context("context-tag"):
                openai.chat.completions.create(model="gpt-4", messages=[], feature_tag="explicit")

        kwargs = mock_track.call_args[1]
        assert kwargs["feature_tag"] == "explicit"

    def test_strips_feature_tag_from_forwarded_params(self):
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 0
        mock_response.usage.completion_tokens = 0
        mock_create = _mock_openai_module.OpenAI.return_value.chat.completions.create
        mock_create.return_value = mock_response

        with patch(TRACK_PATCH_TARGET):
            openai.chat.completions.create(model="gpt-4", messages=[], feature_tag="tag")

        call_kwargs = mock_create.call_args[1]
        assert "feature_tag" not in call_kwargs

    def test_silent_failure_on_track_error(self):
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 0
        mock_response.usage.completion_tokens = 0
        _mock_openai_module.OpenAI.return_value.chat.completions.create.return_value = mock_response

        with patch(TRACK_PATCH_TARGET, side_effect=Exception("fail")):
            response = openai.chat.completions.create(model="gpt-4", messages=[])

        assert response is mock_response


class TestOpenAIStreaming:
    def setup_method(self):
        agentracer.init(tracker_api_key="k", project_id="proj-1", enabled=True)
        _mock_openai_module.OpenAI.return_value.chat.completions.create.reset_mock()

    def test_wraps_stream_and_sends_telemetry(self):
        chunk1 = MagicMock()
        chunk1.usage = None
        chunk2 = MagicMock()
        chunk2.usage = None
        chunk3 = MagicMock()
        chunk3.usage.prompt_tokens = 15
        chunk3.usage.completion_tokens = 25

        _mock_openai_module.OpenAI.return_value.chat.completions.create.return_value = iter([chunk1, chunk2, chunk3])

        with patch(TRACK_PATCH_TARGET) as mock_track:
            stream = openai.chat.completions.create(model="gpt-4", messages=[], stream=True)
            received = list(stream)

        assert len(received) == 3
        mock_track.assert_called_once()
        kwargs = mock_track.call_args[1]
        assert kwargs["input_tokens"] == 15
        assert kwargs["output_tokens"] == 25
        assert kwargs["provider"] == "openai"

    def test_injects_stream_options(self):
        chunk = MagicMock()
        chunk.usage.prompt_tokens = 1
        chunk.usage.completion_tokens = 1
        mock_create = _mock_openai_module.OpenAI.return_value.chat.completions.create
        mock_create.return_value = iter([chunk])

        with patch(TRACK_PATCH_TARGET):
            stream = openai.chat.completions.create(model="gpt-4", messages=[], stream=True)
            list(stream)

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["stream_options"]["include_usage"] is True


class TestTrackedOpenAIConstructor:
    def test_passes_kwargs_to_openai(self):
        client = TrackedOpenAI(api_key="sk-test", base_url="https://custom.api")
        assert client._kwargs == {"api_key": "sk-test", "base_url": "https://custom.api"}
