import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from henry_speech.audio import AudioFormat
from henry_speech.capture.adapters import get_vad_model
from henry_speech.capture.adapters.mlx_silero_vad import SileroVADModel
from henry_speech.capture.config import VADSettings
from henry_speech.synthesis.adapters import get_tts_model
from henry_speech.synthesis.adapters.mlx_chatterbox import MLXChatterboxModel
from henry_speech.synthesis.adapters.piper import PiperModel
from henry_speech.synthesis.config import (
    MLXChatterboxProfile,
    MLXChatterboxSettings,
    PiperProfile,
    PiperSettings,
    TTSProfile,
)
from henry_speech.transcription.adapters import get_stt_model
from henry_speech.transcription.adapters.mlx_parakeet_tdt import ParakeetTDTModel
from henry_speech.transcription.adapters.mlx_qwen3_asr import Qwen3ASRModel
from henry_speech.transcription.adapters.mlx_whisper import WhisperModel
from henry_speech.transcription.config import (
    MLXParakeetTDTProfile,
    MLXParakeetTDTSettings,
    MLXQwen3ASRProfile,
    MLXQwen3ASRSettings,
    MLXWhisperProfile,
    MLXWhisperSettings,
    STTProfile,
)

FORMAT = AudioFormat(sample_rate=16_000, channels=1)


def _install_module(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    **attributes: object,
) -> ModuleType:
    parts = name.split(".")
    for index in range(1, len(parts)):
        package_name = ".".join(parts[:index])
        if package_name not in sys.modules:
            package = ModuleType(package_name)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, package_name, package)

    module = ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _install_loader(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    model: object,
    loaded_ids: list[str],
) -> None:
    def load(model_id: str) -> object:
        loaded_ids.append(model_id)
        return model

    _install_module(monkeypatch, name, load=load)


def test_parakeet_adapter_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def stream_generate(self, _samples, **options):
            assert options == {}
            return iter(
                (SimpleNamespace(text="rozpoznany "), SimpleNamespace(text="tekst"))
            )

    loaded_ids: list[str] = []
    _install_loader(monkeypatch, "mlx_audio.stt.utils", FakeModel(), loaded_ids)
    _install_module(monkeypatch, "mlx.core", array=np.asarray)

    model = ParakeetTDTModel(
        MLXParakeetTDTProfile(model_id="profile/parakeet"),
        MLXParakeetTDTSettings(model_id="settings/parakeet"),
    )
    with model:
        chunks = list(model.transcribe(FORMAT.build_frame(np.zeros(512))))

    assert loaded_ids == ["profile/parakeet"]
    assert [chunk.content for chunk in chunks] == ["rozpoznany ", "tekst"]


def test_qwen3_adapter_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSTTOutput:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeModel:
        def generate(self, *_args, **kwargs):
            assert kwargs == {"stream": False}
            return FakeSTTOutput(" rozpoznany tekst ")

    loaded_ids: list[str] = []
    _install_loader(monkeypatch, "mlx_audio.stt.utils", FakeModel(), loaded_ids)
    _install_module(monkeypatch, "mlx_audio.stt.models.base", STTOutput=FakeSTTOutput)

    model = Qwen3ASRModel(
        MLXQwen3ASRProfile(),
        MLXQwen3ASRSettings(model_id="settings/qwen"),
    )
    with model:
        chunks = list(model.transcribe(FORMAT.build_frame(np.zeros(512))))

    assert loaded_ids == ["settings/qwen"]
    assert [chunk.content for chunk in chunks] == ["rozpoznany tekst"]


def test_whisper_adapter_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    class FakeModel:
        def generate(self, *_args, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text=" transkrypcja ")

    loaded_ids: list[str] = []
    _install_loader(monkeypatch, "mlx_audio.stt.utils", FakeModel(), loaded_ids)

    model = WhisperModel(
        MLXWhisperProfile(model_id="profile/whisper", language="pl"),
        MLXWhisperSettings(model_id="settings/whisper"),
    )
    with model:
        chunks = list(model.transcribe(FORMAT.build_frame(np.zeros(512))))
    assert [chunk.content for chunk in chunks] == ["transkrypcja"]
    assert calls == [{"task": "transcribe", "verbose": None, "language": "pl"}]
    assert loaded_ids == ["profile/whisper"]

    calls.clear()
    model = WhisperModel(MLXWhisperProfile(), MLXWhisperSettings())
    with model:
        list(model.transcribe(FORMAT.build_frame(np.zeros(512))))
    assert calls == [{"task": "transcribe", "verbose": None}]


def test_chatterbox_adapter_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    result = SimpleNamespace(
        audio=np.asarray([0.25, -0.25], dtype=np.float32),
        sample_rate=24_000,
    )
    calls: list[dict] = []

    class FakeModel:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return iter((result,))

    loaded_ids: list[str] = []
    _install_loader(monkeypatch, "mlx_audio.tts.utils", FakeModel(), loaded_ids)

    model = MLXChatterboxModel(
        MLXChatterboxProfile(model_id="profile/chatterbox", lang_code="pl"),
        MLXChatterboxSettings(model_id="settings/chatterbox"),
    )
    with model:
        frames = list(model.synthesize("Dzień dobry."))

    assert loaded_ids == ["profile/chatterbox"]
    assert calls == [{"text": "Dzień dobry.", "verbose": False, "lang_code": "pl"}]
    assert frames[0].format == AudioFormat(sample_rate=24_000, channels=1)
    np.testing.assert_array_equal(frames[0].samples, result.audio)


def test_piper_adapter_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    import henry_speech.synthesis.adapters.piper as piper_module

    downloads: list[tuple[str, str]] = []
    load_calls: list[dict] = []
    synthesis_calls: list[dict] = []
    chunk = SimpleNamespace(
        audio_float_array=np.asarray([0.1, -0.1], dtype=np.float32),
        sample_rate=22_050,
        sample_channels=1,
    )

    class FakeVoice:
        def synthesize(self, **kwargs):
            synthesis_calls.append(kwargs)
            return iter((chunk,))

    def download(*, repo_id: str, filename: str) -> str:
        downloads.append((repo_id, filename))
        return f"/models/{filename}"

    def load_voice(**kwargs) -> FakeVoice:
        load_calls.append(kwargs)
        return FakeVoice()

    monkeypatch.setattr(piper_module, "hf_hub_download", download)
    monkeypatch.setattr(piper_module.PiperVoice, "load", load_voice)

    model = PiperModel(
        PiperProfile(
            voice_path="pl/voice.onnx",
            repo_id="profile/voices",
            speaker_id=2,
            length_scale=1.1,
            noise_scale=0.5,
            noise_w_scale=0.6,
        ),
        PiperSettings(repo_id="settings/voices", normalize_audio=False, volume=0.8),
    )
    with model:
        frames = list(model.synthesize("Dzień dobry."))

    assert downloads == [
        ("profile/voices", "pl/voice.onnx"),
        ("profile/voices", "pl/voice.onnx.json"),
    ]
    assert load_calls == [
        {
            "model_path": "/models/pl/voice.onnx",
            "config_path": "/models/pl/voice.onnx.json",
        }
    ]
    synthesis_config = synthesis_calls[0]["syn_config"]
    assert synthesis_calls[0]["text"] == "Dzień dobry."
    assert synthesis_config.speaker_id == 2
    assert synthesis_config.length_scale == 1.1
    assert synthesis_config.noise_scale == 0.5
    assert synthesis_config.noise_w_scale == 0.6
    assert not synthesis_config.normalize_audio
    assert synthesis_config.volume == 0.8
    assert frames[0].format == AudioFormat(sample_rate=22_050, channels=1)


def test_adapter_factories_validate_selected_profile() -> None:
    tts = get_tts_model(
        TTSProfile(tts={"model_id": "profile/chatterbox"}),
        MLXChatterboxSettings(),
    )
    stt = get_stt_model(
        STTProfile(stt={"language": "pl"}),
        MLXWhisperSettings(),
    )

    assert isinstance(tts, MLXChatterboxModel)
    assert isinstance(stt, WhisperModel)
    assert isinstance(get_vad_model(object(), VADSettings()), SileroVADModel)

    with pytest.raises(ValidationError, match="voice_path"):
        get_tts_model(
            TTSProfile(tts={"voice_path": "voice.onnx"}),
            MLXChatterboxSettings(),
        )
    with pytest.raises(ValidationError, match="language"):
        get_stt_model(
            STTProfile(stt={"language": "pl"}),
            MLXParakeetTDTSettings(),
        )


def test_piper_profile_and_settings_validation() -> None:
    profile = PiperProfile(
        voice_path="voice.onnx",
        speaker_id=2,
        length_scale=1.1,
        noise_scale=0.5,
        noise_w_scale=0.6,
    )
    settings = PiperSettings(normalize_audio=False, volume=0.8)

    assert profile.voice_path == "voice.onnx"
    assert settings.repo_id == "rhasspy/piper-voices"

    with pytest.raises(ValidationError):
        PiperProfile(voice_path="voice.onnx", speaker_id=-1)
    with pytest.raises(ValidationError):
        PiperSettings(volume=0.0)
