from pathlib import Path

import pytest
from pydantic import ValidationError

import henry_resources.local as local_module
from henry_resources import LocalStore
from henry_resources.profiles import Profile
from henry_resources.settings import Settings


def write_profile(root: Path, name: str = "default") -> Path:
    profile = root / "profiles" / name
    prompts = profile / "prompts"
    reactions = profile / "reactions"
    prompts.mkdir(parents=True)
    reactions.mkdir()
    (profile / "profile.yml").write_text(
        """
name: Henry
conversation:
  models:
    fast:
      model_id: ollama:gpt-oss:20b
    detailed:
      model_id: ollama:gpt-oss:20b
  recent_messages: 6
wakeword:
  label: Wakeword
  model_path: wakeword.onnx
tts:
  model_path: voice.onnx
stt:
  model_id: profile/stt
""".strip(),
        encoding="utf-8",
    )
    (prompts / "system.md").write_text(
        "System Polish {conversation_summary}", encoding="utf-8"
    )
    (prompts / "opening.md").write_text(
        "Open Polish {conversation_summary} {recent_conversation}",
        encoding="utf-8",
    )
    (prompts / "summary.md").write_text(
        "Summarize {conversation_summary} {recent_conversation}",
        encoding="utf-8",
    )
    (reactions / "wake.txt").write_text("Tak, słucham.\nJestem.\n", encoding="utf-8")
    (reactions / "wait.txt").write_text(
        "Chwileczkę.\nJuż sprawdzam.\n", encoding="utf-8"
    )
    return profile


def write_settings(root: Path) -> None:
    (root / "settings.yml").write_text(
        """
conversation:
  language_model:
    adapter: langchain
    base_url: http://models.local:11434
speech:
  audio:
    driver: pyaudio
""".strip(),
        encoding="utf-8",
    )


def test_local_store_loads_profile_settings_and_models(tmp_path: Path) -> None:
    profile_path = write_profile(tmp_path)
    write_profile(tmp_path, "second")
    write_settings(tmp_path)
    model_path = tmp_path / "models" / "nested" / "model.onnx"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")

    store = LocalStore(tmp_path)
    profile = store.load_profile("default")

    assert profile.id == "default"
    assert profile.path == profile_path
    assert profile.name == "Henry"
    assert profile.wakeword.label == "Wakeword"
    assert profile.stt == {"model_id": "profile/stt"}
    assert profile.conversation.recent_messages == 6
    assert profile.conversation.prompts.system.startswith("System")
    assert profile.conversation.reactions.wake == ("Tak, słucham.", "Jestem.")
    assert "id" not in profile.model_dump()
    assert [item.id for item in store.list_profiles()] == ["default", "second"]
    inspected = store.inspect_profiles()
    assert [item.id for item in inspected] == ["default", "second"]
    assert all(item.is_valid for item in inspected)
    settings = store.load_settings()
    assert settings.conversation.language_model.adapter == "langchain"
    assert settings.conversation.language_model.base_url == "http://models.local:11434"
    assert settings.speech.audio.driver == "pyaudio"
    assert store.ensure_model_path("nested", "model.onnx") == model_path


def test_local_store_reports_missing_resources(tmp_path: Path) -> None:
    store = LocalStore(tmp_path)
    with pytest.raises(FileNotFoundError, match="Model file does not exist"):
        store.ensure_model_path("missing")
    with pytest.raises(FileNotFoundError, match="Profile directory does not exist"):
        store.load_profile("missing")
    with pytest.raises(FileNotFoundError, match="Profiles directory does not exist"):
        store.list_profiles()
    assert store.inspect_profiles() == []
    with pytest.raises(FileNotFoundError, match="Settings file does not exist"):
        store.load_settings()


def test_profile_requires_fixed_prompt_files_and_valid_configuration(
    tmp_path: Path,
) -> None:
    profile_path = write_profile(tmp_path)
    (profile_path / "prompts" / "opening.md").unlink()
    with pytest.raises(FileNotFoundError, match=r"opening\.md"):
        Profile.load_from_directory(profile_path)

    (profile_path / "prompts" / "opening.md").write_text("open", encoding="utf-8")
    (profile_path / "reactions" / "wait.txt").unlink()
    with pytest.raises(FileNotFoundError, match=r"wait\.txt"):
        Profile.load_from_directory(profile_path)

    (profile_path / "reactions" / "wait.txt").write_text("wait", encoding="utf-8")
    (profile_path / "profile.yml").write_text("name: Henry\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        Profile.load_from_directory(profile_path)


def test_store_root_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HENRY_HOME", str(home))
    assert LocalStore()._root_path == home

    monkeypatch.delenv("HENRY_HOME")
    project = tmp_path / "project"
    nested = project / "a" / "b"
    nested.mkdir(parents=True)
    local = project / ".henry"
    local.mkdir()
    monkeypatch.chdir(nested)
    assert LocalStore()._root_path == local

    local.rmdir()
    fallback = tmp_path / "fallback"
    monkeypatch.setattr(local_module, "user_data_dir", lambda _: str(fallback))
    assert LocalStore()._root_path == fallback


def test_profile_inspection_keeps_invalid_profiles_visible(tmp_path: Path) -> None:
    valid = write_profile(tmp_path)
    invalid = write_profile(tmp_path, "invalid")
    (invalid / "prompts" / "system.md").unlink()
    missing = tmp_path / "profiles" / "missing"
    missing.mkdir()
    malformed = tmp_path / "profiles" / "malformed"
    malformed.mkdir()
    (malformed / "profile.yml").write_text("name: [", encoding="utf-8")
    blank = tmp_path / "profiles" / "blank"
    blank.mkdir()
    (blank / "profile.yml").write_text("name: '   '", encoding="utf-8")
    scalar = tmp_path / "profiles" / "scalar"
    scalar.mkdir()
    (scalar / "profile.yml").write_text("profile", encoding="utf-8")

    entries = {entry.id: entry for entry in LocalStore(tmp_path).inspect_profiles()}
    assert entries[valid.name].is_valid
    assert entries["invalid"].name == "Henry"
    assert "system.md" in entries["invalid"].error
    assert entries["missing"].name == "missing"
    assert entries["malformed"].name == "malformed"
    assert entries["blank"].name == "blank"
    assert entries["scalar"].name == "scalar"


def test_settings_and_profile_validation(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.yml"
    settings_path.write_text("unknown: true\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        Settings.load_from_file(settings_path)

    settings_path.write_text(
        "conversation:\n  adapter: mlx\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="adapter"):
        Settings.load_from_file(settings_path)

    settings_path.write_text(
        "conversation:\n  model:\n    adapter: mlx\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match=r"conversation\.model"):
        Settings.load_from_file(settings_path)

    profile_path = write_profile(tmp_path)
    profile_file = profile_path / "profile.yml"
    profile_file.write_text(
        profile_file.read_text(encoding="utf-8").replace(
            "wakeword.onnx", "wakeword.bin"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="ONNX"):
        Profile.load_from_directory(profile_path)


def test_versioned_default_profile_is_valid() -> None:
    root = Path(__file__).parents[2] / "examples" / "profiles"
    henry = Profile.load_from_directory(root / "default")

    assert henry.name == "Henry"
    assert "comic and controlled" in henry.conversation.prompts.system
    models = henry.conversation.models_mlx
    assert models.fast.model_id == "mlx-community/Qwen3.5-4B-MLX-4bit"
    assert models.detailed.model_id == "mlx-community/Qwen3.5-4B-MLX-4bit"
    assert "mlx" not in henry.conversation.models["fast"]
    assert henry.conversation.reactions.wake
    assert henry.conversation.reactions.wait


def test_versioned_settings_list_all_defaults() -> None:
    path = Path(__file__).parents[2] / "examples" / "settings.yml"

    assert Settings.load_from_file(path) == Settings()
