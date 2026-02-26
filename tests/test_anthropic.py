import sys
from unittest.mock import MagicMock, patch
import agentracer

_mock_anthropic_module = MagicMock()
sys.modules["anthropic"] = _mock_anthropic_module

from agentracer.anthropic import anthropic  # noqa: E402

TRACK_PATCH_TARGET = "agentracer.anthropic.track"


class TestAnthropicNonStreaming:
    def setup_method(self):
        agentracer.init(tracker_api_key="k", project_id="proj-1", enabled=True)
        _mock_anthropic_module.Anthropic.return_value.messages.create.reset_mock()

    def test_sends_telemetry_with_correct_tokens(self):
        mock_response = MagicMock()
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 30
        _mock_anthropic_module.Anthropic.return_value.messages.create.return_value = mock_response

        with patch(TRACK_PATCH_TARGET) as mock_track:
            response = anthropic.messages.create(model="claude-3-opus", max_tokens=100, messages=[])

        assert response is mock_response
        mock_track.assert_called_once()
        kwargs = mock_track.call_args[1]
        assert kwargs["model"] == "claude-3-opus"
        assert kwargs["input_tokens"] == 50
        assert kwargs["output_tokens"] == 30
        assert kwargs["provider"] == "anthropic"

    def test_uses_feature_tag_from_context(self):
        mock_response = MagicMock()
        mock_response.usage.input_tokens = 1
        mock_response.usage.output_tokens = 1
        _mock_anthropic_module.Anthropic.return_value.messages.create.return_value = mock_response

        with patch(TRACK_PATCH_TARGET) as mock_track:
            with agentracer.feature_context("summarizer"):
                anthropic.messages.create(model="claude-3-haiku", max_tokens=50, messages=[])

        kwargs = mock_track.call_args[1]
        assert kwargs["feature_tag"] == "summarizer"

    def test_strips_feature_tag(self):
        mock_response = MagicMock()
        mock_response.usage.input_tokens = 0
        mock_response.usage.output_tokens = 0
        mock_create = _mock_anthropic_module.Anthropic.return_value.messages.create
        mock_create.return_value = mock_response

        with patch(TRACK_PATCH_TARGET):
            anthropic.messages.create(model="claude-3-haiku", max_tokens=50, messages=[], feature_tag="my-tag")

        call_kwargs = mock_create.call_args[1]
        assert "feature_tag" not in call_kwargs

    def test_silent_failure(self):
        mock_response = MagicMock()
        mock_response.usage.input_tokens = 0
        mock_response.usage.output_tokens = 0
        _mock_anthropic_module.Anthropic.return_value.messages.create.return_value = mock_response

        with patch(TRACK_PATCH_TARGET, side_effect=Exception("fail")):
            response = anthropic.messages.create(model="claude-3-haiku", max_tokens=50, messages=[])

        assert response is mock_response


class TestAnthropicStreaming:
    def setup_method(self):
        agentracer.init(tracker_api_key="k", project_id="proj-1", enabled=True)
        _mock_anthropic_module.Anthropic.return_value.messages.create.reset_mock()

    def test_wraps_stream_and_sends_telemetry(self):
        msg_start = MagicMock()
        msg_start.type = "message_start"
        msg_start.message.usage.input_tokens = 100

        delta1 = MagicMock()
        delta1.type = "content_block_delta"

        msg_delta = MagicMock()
        msg_delta.type = "message_delta"
        msg_delta.usage.output_tokens = 40

        msg_stop = MagicMock()
        msg_stop.type = "message_stop"

        events = [msg_start, delta1, msg_delta, msg_stop]
        _mock_anthropic_module.Anthropic.return_value.messages.create.return_value = iter(events)

        with patch(TRACK_PATCH_TARGET) as mock_track:
            stream = anthropic.messages.create(
                model="claude-3-opus", max_tokens=100, messages=[], stream=True
            )
            received = list(stream)

        assert len(received) == 4
        mock_track.assert_called_once()
        kwargs = mock_track.call_args[1]
        assert kwargs["input_tokens"] == 100
        assert kwargs["output_tokens"] == 40
        assert kwargs["provider"] == "anthropic"

    def test_handles_no_usage_events(self):
        delta = MagicMock()
        delta.type = "content_block_delta"
        stop = MagicMock()
        stop.type = "message_stop"

        _mock_anthropic_module.Anthropic.return_value.messages.create.return_value = iter([delta, stop])

        with patch(TRACK_PATCH_TARGET) as mock_track:
            stream = anthropic.messages.create(
                model="claude-3-haiku", max_tokens=50, messages=[], stream=True
            )
            list(stream)

        kwargs = mock_track.call_args[1]
        assert kwargs["input_tokens"] == 0
        assert kwargs["output_tokens"] == 0
