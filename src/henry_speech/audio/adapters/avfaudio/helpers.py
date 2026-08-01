import os
import shutil
from pathlib import Path


def resolve_helper_executable() -> Path:
    configured = os.environ.get("HENRY_AUDIO_HELPER")
    if configured:
        return _require_executable(Path(configured).expanduser())

    adapter_dir = Path(__file__).resolve().parent
    packaged = adapter_dir / "bin" / "henry-audio"
    if packaged.is_file() and os.access(packaged, os.X_OK):
        return packaged

    repository_root = adapter_dir.parents[4]
    development = (
        repository_root
        / "native"
        / "macos"
        / "henry-audio"
        / ".build"
        / "release"
        / "henry-audio"
    )
    if development.is_file() and os.access(development, os.X_OK):
        return development

    installed = shutil.which("henry-audio")
    if installed is not None:
        return Path(installed)

    raise RuntimeError(
        "Native audio helper executable was not found; run "
        "scripts/build-native-audio.sh or set HENRY_AUDIO_HELPER"
    )


def _require_executable(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RuntimeError(f"Native audio helper is not executable: {resolved}")
    return resolved
