import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import hf_hub_download, snapshot_download

from henry_conversation.model.config import LangChainSettings, MLXSettings
from henry_resources import LocalStore, Profile, Settings
from henry_speech.capture.config import MLX_SILERO_VAD_MODEL_ID
from henry_speech.synthesis.config import MLXChatterboxSettings, PiperSettings
from henry_speech.transcription.config import (
    MLXParakeetTDTSettings,
    MLXQwen3ASRSettings,
    MLXWhisperSettings,
)

_OPENWAKEWORD_RELEASE_URL = (
    "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
)
_OPENWAKEWORD_FILES = (
    "embedding_model.onnx",
    "melspectrogram.onnx",
    "silero_vad.onnx",
)


def _exists(path: Path) -> bool:
    return os.path.lexists(path)


def _install_file(
    destination: Path,
    writer: Callable[[Path], None],
) -> bool:
    if _exists(destination):
        print(f"Already exists: {destination}")
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)

    try:
        writer(temporary)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            print(f"Already exists: {destination}")
            return False
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Installed: {destination}")
    return True


def _copy_file(source: Path, destination: Path) -> None:
    def copy(temporary: Path) -> None:
        shutil.copy2(source, temporary)

    _install_file(destination, copy)


def _copy_directory(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if path.is_dir():
            (destination / relative).mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            _copy_file(path, destination / relative)


def _download_openwakeword_file(url: str, destination: Path) -> None:
    def download(temporary: Path) -> None:
        urlretrieve(url, temporary)

    _install_file(destination, download)


def _install_local_data(repository_root: Path, data_root: Path) -> None:
    examples = repository_root / "examples"
    _copy_file(examples / "settings.yml", data_root / "settings.yml")
    _copy_directory(
        examples / "profiles" / "default",
        data_root / "profiles" / "default",
    )
    _copy_file(
        examples / "models" / "openwakeword" / "hey_henry.onnx",
        data_root / "models" / "openwakeword" / "hey_henry.onnx",
    )


def _install_openwakeword_models(data_root: Path) -> None:
    destination = data_root / "models" / "openwakeword"
    for filename in _OPENWAKEWORD_FILES:
        _download_openwakeword_file(
            f"{_OPENWAKEWORD_RELEASE_URL}/{filename}",
            destination / filename,
        )


def _conversation_models(profile: Profile, settings: Settings) -> tuple[str, ...]:
    match settings.conversation.language_model:
        case MLXSettings():
            models = profile.conversation.models_mlx
            selected = [models.fast.model_id, models.detailed.model_id]
            if settings.conversation.classify_ambiguous:
                if models.classifier is None:
                    raise ValueError(
                        "Ambiguous-turn classification requires a classifier model"
                    )
                selected.append(models.classifier.model_id)
            return tuple(selected)
        case LangChainSettings():
            models = profile.conversation.models_langchain
            if settings.conversation.classify_ambiguous and models.classifier is None:
                raise ValueError(
                    "Ambiguous-turn classification requires a classifier model"
                )
            return ()


def _stt_model(profile: Profile, settings: Settings) -> str:
    selected = settings.speech.stt
    match selected:
        case MLXParakeetTDTSettings():
            configured = profile.stt_mlx_parakeet_tdt.model_id
        case MLXQwen3ASRSettings():
            configured = profile.stt_mlx_qwen3_asr.model_id
        case MLXWhisperSettings():
            configured = profile.stt_mlx_whisper.model_id
    return configured or selected.model_id


def _tts_models(
    profile: Profile,
    settings: Settings,
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    selected = settings.speech.tts
    match selected:
        case PiperSettings():
            configured = profile.tts_piper
            repository = configured.repo_id or selected.repo_id
            filename = configured.model_path
            return (), ((repository, filename), (repository, filename + ".json"))
        case MLXChatterboxSettings():
            configured = profile.tts_mlx_chatterbox
            return (configured.model_id or selected.model_id,), ()


def _hugging_face_models(
    profile: Profile,
    settings: Settings,
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    repositories = list(_conversation_models(profile, settings))
    if settings.speech.vad.adapter == "mlx:silero_vad":
        repositories.append(MLX_SILERO_VAD_MODEL_ID)
    repositories.append(_stt_model(profile, settings))
    tts_repositories, files = _tts_models(profile, settings)
    repositories.extend(tts_repositories)
    return tuple(dict.fromkeys(repositories)), tuple(dict.fromkeys(files))


def _download_hugging_face_models(profile: Profile, settings: Settings) -> None:
    repositories, files = _hugging_face_models(profile, settings)
    for repository in repositories:
        if Path(repository).expanduser().exists():
            print(f"Local model exists: {repository}")
            continue
        print(f"Downloading Hugging Face model: {repository}")
        snapshot_download(repo_id=repository)

    for repository, filename in files:
        print(f"Downloading Hugging Face file: {repository}/{filename}")
        hf_hub_download(repo_id=repository, filename=filename)


def install(
    repository_root: Path,
    data_root: Path | None = None,
) -> Path:
    repository_root = repository_root.resolve()
    if data_root is None:
        configured_root = os.getenv("HENRY_HOME")
        data_root = (
            Path(configured_root).expanduser()
            if configured_root
            else repository_root / ".henry"
        )
    data_root = data_root.resolve()

    print(f"Initializing Henry data: {data_root}")
    _install_local_data(repository_root, data_root)

    store = LocalStore(data_root)
    settings = store.load_settings()
    profile = store.load_profile("default")

    _install_openwakeword_models(data_root)
    _download_hugging_face_models(profile, settings)
    print(f"Henry is ready: {data_root}")
    return data_root


def main() -> None:
    install(Path(__file__).parents[1])


if __name__ == "__main__":
    main()
