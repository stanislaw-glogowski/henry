from dataclasses import replace
from pathlib import Path

from henry_cli.events import ReplyDraft, TelemetrySnapshot, TranscriptionDraft
from henry_cli.ui.state import (
    AssistantMessage,
    ConversationState,
    PhraseState,
    RuntimeInfo,
    RuntimeMode,
    State,
    UserMessage,
)
from henry_common.events import ShutdownEvent
from henry_conversation import (
    CancelReply,
    ConversationReady,
    ReplyDraftUpdated,
    ReplyGenerationCompleted,
    ReplyGenerationStarted,
    ReplyPhrase,
)
from henry_resources import Profile, Settings
from henry_speech.audio import AudioDevice, AudioDevices
from henry_speech.events import (
    AudioDevicesSelected,
    ReplyPhraseDelivered,
    ReplyPhrasePlaybackStarted,
    SpeechReady,
    UserTurnCommitted,
    VoiceSessionMode,
    VoiceSessionModeChanged,
)


def profile(**updates) -> Profile:
    values = {
        "id": "test",
        "path": Path("/profiles/test"),
        "name": "Test Henry",
        "conversation": {
            "models": {
                "fast": {"model_id": "test/fast"},
                "detailed": {"model_id": "test/detailed"},
                "classifier": {"model_id": "test/fast"},
            },
            "prompts": {
                "system": "system {conversation_summary}",
                "opening": "opening {conversation_summary}",
                "summary": "summary {conversation_summary} {recent_conversation}",
            },
        },
        "wakeword": {
            "label": "Wake",
            "model_path": "wake.onnx",
            "threshold": 0.7,
        },
        "stt": {},
        "tts": {"model_path": "voice.onnx"},
    }
    values.update(updates)
    return Profile.model_validate(values)


def test_runtime_info_maps_all_runtime_adapter_variants() -> None:
    value = profile()
    settings = Settings.model_validate(
        {
            "conversation": {"language_model": {"adapter": "langchain"}},
            "speech": {"stt": {"adapter": "mlx:parakeet-tdt"}},
        }
    )
    info = RuntimeInfo.from_runtime(value, settings)
    assert info.profile_id == "test"
    assert info.wakeword_label == "Wake"
    assert info.llm_model == "test/fast · test/detailed"
    assert info.stt_model == settings.speech.stt.model_id
    assert info.tts_model.endswith("/voice.onnx")

    for adapter, profile_stt, expected in (
        ("mlx:qwen3-asr", {"model_id": "profile/qwen"}, "profile/qwen"),
        ("mlx:whisper", {"model_id": "profile/whisper"}, "profile/whisper"),
    ):
        variant = profile(stt=profile_stt)
        variant_settings = Settings.model_validate(
            {
                "conversation": {"language_model": {"adapter": "mlx"}},
                "speech": {"stt": {"adapter": adapter}},
            }
        )
        assert RuntimeInfo.from_runtime(variant, variant_settings).stt_model == expected

    chatterbox = profile(tts={"model_id": "profile/chatterbox"})
    chatterbox_settings = Settings.model_validate(
        {
            "conversation": {"language_model": {"adapter": "mlx"}},
            "speech": {"tts": {"adapter": "mlx:chatterbox"}},
        }
    )
    chatterbox_info = RuntimeInfo.from_runtime(chatterbox, chatterbox_settings)
    assert chatterbox_info.tts_model == "profile/chatterbox"
    assert chatterbox_info.llm_adapter == "mlx"


def test_conversation_state_tracks_drafts_phrases_and_interruption() -> None:
    state = ConversationState()
    assert state.update_transcription(None) is state
    state = state.update_transcription(TranscriptionDraft(1, "Hel", False))
    state = state.update_transcription(TranscriptionDraft(1, "Hello", True))
    assert state.messages == (UserMessage(1, "Hello"),)
    state = state.commit_user(1, "Hello!")
    assert state.messages == (UserMessage(1, "Hello!", committed=True),)
    state = state.commit_user(2, "Second")
    state = state.update_transcription(TranscriptionDraft(3, "Live", False))
    state = state.update_transcription(TranscriptionDraft(4, "Replacement", False))
    assert not any(
        isinstance(message, UserMessage) and message.turn_id == 3
        for message in state.messages
    )

    assert state.update_reply_draft(None) is state
    state = state.update_reply_draft(ReplyDraft(10, "Building"))
    state = state.update_reply_draft(ReplyDraft(10, "Building now"))
    assistant = state.messages[-1]
    assert isinstance(assistant, AssistantMessage)
    assert assistant.draft == "Building now"

    state = state.queue_phrase(10, 1, "First sentence.")
    assert state.queue_phrase(10, 1, "duplicate") is state
    state = state.queue_phrase(11, 1, "New reply.")
    state = state.transition_phrase(10, 1, PhraseState.SPEAKING)
    state = state.transition_phrase(10, 1, PhraseState.QUEUED)
    state = state.transition_phrase(10, 1, PhraseState.DELIVERED)
    state = state.queue_phrase(10, 2, "Queued sentence.")
    state = state.queue_phrase(10, 3, "Speaking sentence.")
    state = state.transition_phrase(10, 3, PhraseState.SPEAKING)
    state = state.transition_phrase(999, 1, PhraseState.DELIVERED)
    state = state.complete_reply(10)
    state = state.complete_reply(999)

    state = state.update_reply_draft(ReplyDraft(10, "unfinished draft"))
    state = state.interrupt_reply(10)
    interrupted = next(
        message
        for message in state.messages
        if isinstance(message, AssistantMessage) and message.reply_id == 10
    )
    assert interrupted.interrupted
    assert interrupted.draft == "unfinished draft"
    assert [phrase.text for phrase in interrupted.phrases] == [
        "First sentence.",
        "Queued sentence.",
        "Speaking sentence.",
    ]
    assert [phrase.state for phrase in interrupted.phrases] == [
        PhraseState.DELIVERED,
        PhraseState.QUEUED,
        PhraseState.SPEAKING,
    ]
    assert state.update_reply_draft(ReplyDraft(10, "ignored")) is state
    assert state.queue_phrase(10, 2, "ignored") is state
    assert state.transition_phrase(10, 1, PhraseState.QUEUED) is state
    assert state.complete_reply(10) is state
    assert state.interrupt_reply(999) is state
    assert state.interrupt_reply(None).messages[-1].interrupted


def test_conversation_state_bounds_history() -> None:
    state = ConversationState()
    for turn_id in range(205):
        state = state.commit_user(turn_id, str(turn_id))
    assert len(state.messages) == 200
    assert state.messages[0] == UserMessage(5, "5", committed=True)


def test_state_reduces_runtime_events_and_reports_modes() -> None:
    state = State()
    assert state.mode is RuntimeMode.STARTING
    state = state.with_runtime(profile(), Settings())
    state = state.reduce_event(ConversationReady())
    state = state.reduce_event(SpeechReady())
    assert state.is_ready
    assert state.mode is RuntimeMode.WAITING

    state = state.reduce_event(
        AudioDevicesSelected(
            "test-driver",
            AudioDevices(AudioDevice("Mic"), AudioDevice("Speakers")),
        )
    )
    assert state.info.input_device == "Mic"
    assert state.info.output_device == "Speakers"

    state = state.reduce_event(VoiceSessionModeChanged(VoiceSessionMode.ACTIVE))
    assert state.mode is RuntimeMode.LISTENING
    telemetry = TelemetrySnapshot(
        transcription=TranscriptionDraft(1, "Question", False),
        reply=ReplyDraft(1, "Draft"),
    )
    state = state.with_telemetry(telemetry)
    assert state.mode is RuntimeMode.TRANSCRIBING
    state = state.reduce_event(UserTurnCommitted(1, "Question?"))
    state = state.reduce_event(ReplyGenerationStarted(1))
    assert state.mode is RuntimeMode.THINKING
    state = state.reduce_event(ReplyPhrase(1, 1, "Answer."))
    state = state.reduce_event(ReplyPhrasePlaybackStarted(1, 1))
    assert state.mode is RuntimeMode.SPEAKING
    state = state.reduce_event(ReplyPhraseDelivered(1, 1))
    state = state.reduce_event(ReplyGenerationCompleted(1))
    assert state.generating_replies == frozenset()

    state = state.reduce_event(ReplyGenerationStarted(2))
    state = state.reduce_event(ReplyDraftUpdated(2, "Interrupted draft"))
    state = state.reduce_event(CancelReply("", 2))
    interrupted = state.conversation.messages[-1]
    assert isinstance(interrupted, AssistantMessage)
    assert interrupted.draft == "Interrupted draft"
    state = state.reduce_event(ReplyGenerationStarted(3))
    state = state.reduce_event(CancelReply())
    assert state.generating_replies == frozenset()
    state = state.reduce_event(ShutdownEvent())
    assert state.mode is RuntimeMode.SHUTTING_DOWN
    assert state.reduce_event(replace(ShutdownEvent())) == state
