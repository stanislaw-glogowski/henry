from pathlib import Path

from henry_resources.locator import locate_root
from henry_resources.models import ensure_model_path


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

    assert ensure_model_path("wake", "model.onnx") == model
