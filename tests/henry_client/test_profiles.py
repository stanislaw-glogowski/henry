import pytest

from henry_client.profiles import Profile, ProfileKind


@pytest.mark.parametrize("kind", list(ProfileKind))
def test_profile_builds_prompt_for_selected_kind(kind: ProfileKind) -> None:
    profile = Profile.build(
        name="Henry",
        voice_model="voice.onnx",
        wakeword_model="wake.onnx",
        wakeword_reply="Ready.",
        system_language="Polish",
        kind=kind,
    )

    assert profile.system_prompt is not None
    assert "Henry" in profile.system_prompt
    assert "Polish" in profile.system_prompt


def test_profile_preserves_explicit_prompt() -> None:
    profile = Profile.build(
        name="Henry",
        voice_model="voice.onnx",
        wakeword_model="wake.onnx",
        wakeword_reply="Ready.",
        system_language="Polish",
        system_prompt="custom",
    )

    assert profile.system_prompt == "custom"
