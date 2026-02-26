import sys
from unittest.mock import MagicMock, patch
import agentracer

_mock_genai_module = MagicMock()
_mock_google = MagicMock()
_mock_google.generativeai = _mock_genai_module
sys.modules["google"] = _mock_google
sys.modules["google.generativeai"] = _mock_genai_module

from agentracer.gemini import gemini  # noqa: E402

TRACK_PATCH_TARGET = "agentracer.gemini.track"


class TestGeminiNonStreaming:
    def setup_method(self):
        agentracer.init(tracker_api_key="k", project_id="proj-1", enabled=True)
        _mock_genai_module.GenerativeModel.return_value.generate_content.reset_mock()

    def test_sends_telemetry_with_correct_tokens(self):
        mock_response = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 20
        mock_response.usage_metadata.candidates_token_count = 35
        _mock_genai_module.GenerativeModel.return_value.generate_content.return_value = mock_response

        with patch(TRACK_PATCH_TARGET) as mock_track:
            model = gemini.GenerativeModel("gemini-pro")
            response = model.generate_content("Hello")

        assert response is mock_response
        mock_track.assert_called_once()
        kwargs = mock_track.call_args[1]
        assert kwargs["model"] == "gemini-pro"
        assert kwargs["input_tokens"] == 20
        assert kwargs["output_tokens"] == 35
        assert kwargs["provider"] == "gemini"

    def test_uses_feature_tag_from_context(self):
        mock_response = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 1
        mock_response.usage_metadata.candidates_token_count = 1
        _mock_genai_module.GenerativeModel.return_value.generate_content.return_value = mock_response

        with patch(TRACK_PATCH_TARGET) as mock_track:
            model = gemini.GenerativeModel("gemini-pro")
            with agentracer.feature_context("embed"):
                model.generate_content("test")

        kwargs = mock_track.call_args[1]
        assert kwargs["feature_tag"] == "embed"

    def test_explicit_feature_tag(self):
        mock_response = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 1
        mock_response.usage_metadata.candidates_token_count = 1
        _mock_genai_module.GenerativeModel.return_value.generate_content.return_value = mock_response

        with patch(TRACK_PATCH_TARGET) as mock_track:
            model = gemini.GenerativeModel("gemini-pro")
            model.generate_content("test", feature_tag="my-tag")

        kwargs = mock_track.call_args[1]
        assert kwargs["feature_tag"] == "my-tag"

    def test_silent_failure(self):
        mock_response = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 0
        mock_response.usage_metadata.candidates_token_count = 0
        _mock_genai_module.GenerativeModel.return_value.generate_content.return_value = mock_response

        with patch(TRACK_PATCH_TARGET, side_effect=Exception("fail")):
            model = gemini.GenerativeModel("gemini-pro")
            response = model.generate_content("test")

        assert response is mock_response


class TestGeminiStreaming:
    def setup_method(self):
        agentracer.init(tracker_api_key="k", project_id="proj-1", enabled=True)
        _mock_genai_module.GenerativeModel.return_value.generate_content.reset_mock()

    def test_wraps_stream_and_sends_telemetry(self):
        chunk1 = MagicMock()
        chunk1.usage_metadata.prompt_token_count = 10
        chunk1.usage_metadata.candidates_token_count = 5

        chunk2 = MagicMock()
        chunk2.usage_metadata.prompt_token_count = 10
        chunk2.usage_metadata.candidates_token_count = 12

        chunk3 = MagicMock()
        chunk3.usage_metadata.prompt_token_count = 10
        chunk3.usage_metadata.candidates_token_count = 18

        _mock_genai_module.GenerativeModel.return_value.generate_content.return_value = iter([chunk1, chunk2, chunk3])

        with patch(TRACK_PATCH_TARGET) as mock_track:
            model = gemini.GenerativeModel("gemini-pro")
            stream = model.generate_content("Hello", stream=True)
            received = list(stream)

        assert len(received) == 3
        mock_track.assert_called_once()
        kwargs = mock_track.call_args[1]
        assert kwargs["input_tokens"] == 10
        assert kwargs["output_tokens"] == 18
        assert kwargs["provider"] == "gemini"
        assert kwargs["model"] == "gemini-pro"
