from pathlib import Path

import pytest

from henry_common import storage
from henry_common.storage import locate_root
from henry_common.models import ensure_file_path


def test_locate_root_prefers_environment(monkeypatch, tmp_path: Path) -> None:
    configured = tmp_path / "configured"
    monkeypatch.setenv("HENRY_HOME", str(configured))

    assert locate_root() == configured


def test_locate_root_finds_parent_local_directory(monkeypatch, tmp_path: Path) -> None:
    local = tmp_path / ".henry"
    nested = tmp_path / "one" / "two"
    local.mkdir()
    nested.mkdir(parents=True)
    monkeypatch.delenv("HENRY_HOME", raising=False)
    monkeypatch.chdir(nested)

    assert locate_root() == local


def test_ensure_model_path_requires_existing_file(monkeypatch, tmp_path: Path) -> None:
    model = tmp_path / "models" / "wake" / "model.onnx"
    model.parent.mkdir(parents=True)
    model.touch()
    monkeypatch.setenv("HENRY_HOME", str(tmp_path))

    assert ensure_file_path("wake", "model.onnx") == model


def test_locate_root_falls_back_to_platform_data_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fallback = tmp_path / "application-support"
    monkeypatch.delenv("HENRY_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(storage, "user_data_dir", lambda _: str(fallback))

    assert locate_root() == fallback


def test_ensure_model_path_rejects_missing_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HENRY_HOME", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="missing.onnx"):
        ensure_file_path("wake", "missing.onnx")
