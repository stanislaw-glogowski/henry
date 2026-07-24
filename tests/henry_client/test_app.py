import asyncio
import sys
from dataclasses import dataclass
from types import ModuleType

import pytest

from henry_client import app as app_module
from henry_client.app import App, AppConfig
from henry_client.config import VADConfig, WakeWordConfig
from henry_client.profiles import Profile, ProfileKind
from tests.support import RecordingEventSink


def profile() -> Profile:
    return Profile.build(
        kind=ProfileKind.DEFAULT,
        name="Henry",
        system_language="Polish",
        wakeword_reply="Ready.",
        wakeword_model="profile-wakeword.onnx",
        voice_model="voice.onnx",
    )


@pytest.mark.parametrize(
    ("wakeword", "expected_model", "expected_reply"),
    [
        (WakeWordConfig(), "profile-wakeword.onnx", "Ready."),
        (
            WakeWordConfig(
                model_path="override-wakeword.onnx",
                reply_message="Override.",
            ),
            "override-wakeword.onnx",
            "Override.",
        ),
    ],
)
def test_app_composes_services_and_forwards_configuration(
    monkeypatch,
    wakeword: WakeWordConfig,
    expected_model: str,
    expected_reply: str,
) -> None:
    recorded = {}

    class FakePyAudioSession:
        def __enter__(self):
            recorded["audio_session"] = self
            return self

        def __exit__(self, *_):
            recorded["audio_session_closed"] = True

    class FakePyAudioStream:
        @staticmethod
        def input(session):
            recorded["input_session"] = session
            return "input"

        @staticmethod
        def output(session):
            recorded["output_session"] = session
            return "output"

    class FakeWakeWordModel:
        def __init__(self, model_path):
            recorded["wakeword_model"] = model_path

    class FakeVADModel:
        def __init__(self):
            recorded["vad_model"] = self

    class FakeSTTModel:
        def __init__(self):
            recorded["stt_model"] = self

    class FakeTTSModel:
        def __init__(self, model_path):
            recorded["tts_model"] = model_path

    @dataclass
    class FakeResponderConfig:
        model_id: str
        system_prompt: str
        activation_text: str

    class FakeResponder:
        def __init__(self, config):
            recorded["responder_config"] = config

    class FakeService:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            recorded.setdefault("services", []).append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

    class FakeOrchestrator:
        def __init__(self, **kwargs):
            recorded["orchestrator"] = kwargs

        async def run(self, shutdown):
            recorded["shutdown"] = shutdown

    audio_adapters = ModuleType("henry_client.audio.adapters")
    audio_adapters.OpenWakeWordModel = FakeWakeWordModel
    audio_adapters.PyAudioSession = FakePyAudioSession
    audio_adapters.PyAudioStream = FakePyAudioStream
    audio_adapters.SileroVADModel = FakeVADModel

    speech_adapters = ModuleType("henry_client.speech.adapters")
    speech_adapters.ParakeetSTTModel = FakeSTTModel
    speech_adapters.PiperTTSModel = FakeTTSModel

    reply_adapters = ModuleType("henry_client.reply.adapters")
    reply_adapters.__path__ = []
    reply_adapter = ModuleType("henry_client.reply.adapters.mlx_lm")
    reply_adapter.MLXResponder = FakeResponder
    reply_adapter.MLXResponderConfig = FakeResponderConfig

    monkeypatch.setitem(sys.modules, "henry_client.audio.adapters", audio_adapters)
    monkeypatch.setitem(sys.modules, "henry_client.speech.adapters", speech_adapters)
    monkeypatch.setitem(sys.modules, "henry_client.reply.adapters", reply_adapters)
    monkeypatch.setitem(
        sys.modules,
        "henry_client.reply.adapters.mlx_lm",
        reply_adapter,
    )
    monkeypatch.setattr(app_module, "AudioService", FakeService)
    monkeypatch.setattr(app_module, "SpeechService", FakeService)
    monkeypatch.setattr(app_module, "ReplyService", FakeService)
    monkeypatch.setattr(app_module, "Orchestrator", FakeOrchestrator)

    events = RecordingEventSink()
    vad = VADConfig(threshold=0.4)
    shutdown = asyncio.Event()
    config = AppConfig(
        profile=profile(),
        language_model="local/model",
        vad=vad,
        wakeword=wakeword,
        max_empty_segments=5,
    )

    asyncio.run(App(config=config, events=events).run(shutdown))

    assert recorded["wakeword_model"] == expected_model
    assert recorded["tts_model"] == "voice.onnx"
    assert recorded["responder_config"].model_id == "local/model"
    assert recorded["responder_config"].activation_text == expected_reply
    assert recorded["audio_session_closed"]
    assert recorded["orchestrator"] == {
        "audio": recorded["services"][0],
        "speech": recorded["services"][1],
        "reply": recorded["services"][2],
        "events": events,
        "vad_config": vad,
        "wakeword_config": wakeword,
        "max_empty_segments": 5,
    }
    assert recorded["shutdown"] is shutdown
