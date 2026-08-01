import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from henry_speech.audio import AudioFormat
from henry_speech.synthesis.adapters.mlx_audio.chatterbox import ChatterboxTTSModel
from henry_speech.synthesis.config import TTSProfile
from henry_speech.transcription.adapters.mlx_audio.qwen3_asr import Qwen3ASRModel
from henry_speech.transcription.adapters.mlx_audio.whisper import WhisperModel
from henry_speech.transcription.config import STTProfile

FORMAT = AudioFormat(sample_rate=16_000, channels=1)


def _install_fake_mlx_loader(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model_module_name: str,
    loader_module_name: str,
    loader_name: str,
    model: object,
) -> None:
    module_names = {model_module_name, loader_module_name}
    package_names = {
        package_name
        for module_name in module_names
        for index in range(1, len(module_name.split(".")))
        if (package_name := ".".join(module_name.split(".")[:index]))
    }

    for package_name in sorted(package_names, key=lambda name: name.count(".")):
        package = ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    model_module = ModuleType(model_module_name)
    model_module.Model = type(model)
    monkeypatch.setitem(sys.modules, model_module_name, model_module)

    loader_module = ModuleType(loader_module_name)
    setattr(loader_module, loader_name, lambda _model_id: model)
    monkeypatch.setitem(sys.modules, loader_module_name, loader_module)


def test_qwen3_adapter_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def generate(self, *_args, **_kwargs):
            return SimpleNamespace(text=" rozpoznany tekst ")

    _install_fake_mlx_loader(
        monkeypatch,
        model_module_name="mlx_audio.stt.models.qwen3_asr",
        loader_module_name="mlx_audio.stt.utils",
        loader_name="load_model",
        model=FakeModel(),
    )
    model = Qwen3ASRModel(STTProfile(model="qwen"))
    with model:
        chunks = list(model.transcribe(FORMAT.build_frame(np.zeros(512))))
    assert [chunk.content for chunk in chunks] == ["rozpoznany tekst"]


def test_whisper_adapter_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    class FakeModel:
        def generate(self, *_args, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text=" transkrypcja ")

    _install_fake_mlx_loader(
        monkeypatch,
        model_module_name="mlx_audio.stt.models.whisper",
        loader_module_name="mlx_audio.stt.utils",
        loader_name="load_model",
        model=FakeModel(),
    )
    model = WhisperModel(STTProfile(model="whisper", language="pl"))
    with model:
        chunks = list(model.transcribe(FORMAT.build_frame(np.zeros(512))))
    assert [chunk.content for chunk in chunks] == ["transkrypcja"]
    assert calls == [{"task": "transcribe", "verbose": None, "language": "pl"}]

    calls.clear()
    model = WhisperModel(STTProfile(model="whisper"))
    with model:
        list(model.transcribe(FORMAT.build_frame(np.zeros(512))))
    assert calls == [{"task": "transcribe", "verbose": None}]


def test_chatterbox_adapter_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    result = SimpleNamespace(
        audio=np.asarray([0.25, -0.25], dtype=np.float32),
        sample_rate=24_000,
    )

    class FakeModel:
        def generate(self, **_kwargs):
            return iter((result,))

    _install_fake_mlx_loader(
        monkeypatch,
        model_module_name="mlx_audio.tts.models.chatterbox",
        loader_module_name="mlx_audio.tts.utils",
        loader_name="load",
        model=FakeModel(),
    )
    model = ChatterboxTTSModel(TTSProfile(model="chatterbox"))
    with model:
        frames = list(model.synthesize("Dzień dobry."))
    assert frames[0].format == AudioFormat(sample_rate=24_000, channels=1)
    np.testing.assert_array_equal(frames[0].samples, result.audio)
