from unittest.mock import patch, MagicMock
import agentracer


class TestInit:
    def test_sets_config(self):
        agentracer.init(
            tracker_api_key="key-123",
            project_id="proj-1",
        )
        assert agentracer._config["tracker_api_key"] == "key-123"
        assert agentracer._config["project_id"] == "proj-1"
        assert agentracer._config["environment"] == "production"
        assert agentracer._config["host"] == "https://api.agentracer.dev"

    def test_sets_custom_options(self):
        agentracer.init(
            tracker_api_key="k",
            project_id="p",
            environment="staging",
            debug=True,
            enabled=False,
        )
        assert agentracer._config["environment"] == "staging"
        assert agentracer._config["debug"] is True
        assert agentracer._config["enabled"] is False


class TestObserve:
    def test_sets_feature_tag_sync(self):
        captured = {}

        @agentracer.observe(feature_tag="chatbot")
        def my_func():
            captured["tag"] = agentracer._current_feature_tag.get()
            return 42

        result = my_func()
        assert result == 42
        assert captured["tag"] == "chatbot"

    def test_feature_tag_resets_after_call(self):
        @agentracer.observe(feature_tag="temp")
        def my_func():
            pass

        my_func()
        assert agentracer._current_feature_tag.get() is None


class TestFeatureContext:
    def test_sets_tag_in_context(self):
        with agentracer.feature_context("search"):
            assert agentracer._current_feature_tag.get() == "search"
        assert agentracer._current_feature_tag.get() is None


class TestTrack:
    @patch("agentracer._send_telemetry_sync")
    def test_sends_correct_payload(self, mock_send):
        agentracer.init(tracker_api_key="k", project_id="proj-1", enabled=True)
        # Force sync path by running outside async loop
        agentracer.track(
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            latency_ms=200,
            feature_tag="chat",
            provider="openai",
        )
        # track uses threading for sync, give it time
        import time
        time.sleep(0.1)
        mock_send.assert_called_once()
        payload = mock_send.call_args[0][0]
        assert payload["project_id"] == "proj-1"
        assert payload["provider"] == "openai"
        assert payload["model"] == "gpt-4"
        assert payload["feature_tag"] == "chat"
        assert payload["input_tokens"] == 100
        assert payload["output_tokens"] == 50

    def test_skips_when_disabled(self):
        agentracer.init(tracker_api_key="k", project_id="p", enabled=False)
        with patch("agentracer._send_telemetry_sync") as mock_send:
            agentracer.track(model="m", input_tokens=0, output_tokens=0, latency_ms=0)
            import time
            time.sleep(0.1)
            mock_send.assert_not_called()

    @patch("agentracer._send_telemetry_sync")
    def test_picks_up_context_var_feature_tag(self, mock_send):
        agentracer.init(tracker_api_key="k", project_id="proj-1", enabled=True)
        with agentracer.feature_context("from-context"):
            agentracer.track(
                model="gpt-4",
                input_tokens=10,
                output_tokens=5,
                latency_ms=100,
                provider="openai",
            )
        import time
        time.sleep(0.1)
        mock_send.assert_called_once()
        payload = mock_send.call_args[0][0]
        assert payload["feature_tag"] == "from-context"

    @patch("agentracer._send_telemetry_sync")
    def test_explicit_feature_tag_overrides_context(self, mock_send):
        agentracer.init(tracker_api_key="k", project_id="proj-1", enabled=True)
        with agentracer.feature_context("from-context"):
            agentracer.track(
                model="gpt-4",
                input_tokens=10,
                output_tokens=5,
                latency_ms=100,
                feature_tag="explicit",
                provider="openai",
            )
        import time
        time.sleep(0.1)
        mock_send.assert_called_once()
        payload = mock_send.call_args[0][0]
        assert payload["feature_tag"] == "explicit"

    @patch("agentracer._send_telemetry_sync")
    def test_defaults_to_unknown_without_context(self, mock_send):
        agentracer.init(tracker_api_key="k", project_id="proj-1", enabled=True)
        agentracer.track(
            model="gpt-4",
            input_tokens=10,
            output_tokens=5,
            latency_ms=100,
        )
        import time
        time.sleep(0.1)
        mock_send.assert_called_once()
        payload = mock_send.call_args[0][0]
        assert payload["feature_tag"] == "unknown"


class TestAgentRun:
    @patch("agentracer._send_run_api_sync")
    def test_context_manager_sends_start_and_end(self, mock_api):
        agentracer.init(tracker_api_key="k", project_id="proj-1", enabled=True)
        with agentracer.AgentRun(run_name="test-run", feature_tag="bot") as run:
            assert run.run_id is not None
            assert agentracer._current_run.get() is run
        # After exit, current_run should be reset
        assert agentracer._current_run.get() is None
        import time
        time.sleep(0.1)
        # Should have called start and end
        calls = mock_api.call_args_list
        assert len(calls) >= 2
        start_payload = calls[0][0][0]
        assert start_payload["run_name"] == "test-run"
        end_payload = calls[1][0][0]
        assert end_payload["status"] == "completed"
