from dataclasses import dataclass, field, replace
from enum import Enum, auto

from henry_common.events import Event, ShutdownEvent
from henry_conversation.events import (
    CancelReply,
    ConversationReady,
    ReplyDraftUpdated,
    ReplyGenerationCompleted,
    ReplyGenerationStarted,
    ReplyId,
    ReplyPhrase,
)
from henry_resources import Profile, Settings
from henry_speech.events import (
    AudioDevicesSelected,
    ReplyPhraseDelivered,
    ReplyPhrasePlaybackStarted,
    SpeechReady,
    UserTurnCommitted,
    VoiceSessionMode,
    VoiceSessionModeChanged,
)
from henry_speech.synthesis.config import MLXChatterboxSettings, PiperSettings

from ..events import ReplyDraft, TelemetrySnapshot, TranscriptionDraft


class PhraseState(Enum):
    QUEUED = auto()
    SPEAKING = auto()
    DELIVERED = auto()


class RuntimeMode(Enum):
    STARTING = "STARTING"
    WAITING = "WAITING"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    SHUTTING_DOWN = "SHUTTING DOWN"


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    profile_id: str = ""
    profile_name: str = "No profile"
    audio_driver: str = "—"
    input_device: str = "Waiting…"
    output_device: str = "Waiting…"
    vad_adapter: str = "—"
    vad_threshold: float = 0.0
    wakeword_model: str = "—"
    wakeword_threshold: float = 0.0
    stt_adapter: str = "—"
    stt_model: str = "—"
    llm_adapter: str = "—"
    llm_model: str = "—"
    tts_adapter: str = "—"
    tts_model: str = "—"

    @classmethod
    def from_runtime(cls, profile: Profile, settings: Settings) -> RuntimeInfo:
        speech = settings.speech
        conversation_adapter = settings.conversation.language_model.adapter
        if conversation_adapter == "mlx":
            models = profile.conversation.models_mlx
        else:
            models = profile.conversation.models_langchain
        model_ids = tuple(
            dict.fromkeys(
                model.model_id
                for model in (models.fast, models.detailed, models.classifier)
                if model is not None
            )
        )

        stt_adapter = speech.stt.adapter
        match stt_adapter:
            case "mlx:parakeet-tdt":
                stt_model = profile.stt_mlx_parakeet_tdt.model_id or speech.stt.model_id
            case "mlx:qwen3-asr":
                stt_model = profile.stt_mlx_qwen3_asr.model_id or speech.stt.model_id
            case "mlx:whisper":
                stt_model = profile.stt_mlx_whisper.model_id or speech.stt.model_id

        match speech.tts:
            case PiperSettings() as tts_settings:
                piper = profile.tts_piper
                tts_model = (
                    f"{piper.repo_id or tts_settings.repo_id}/{piper.model_path}"
                )
            case MLXChatterboxSettings() as tts_settings:
                tts_model = profile.tts_mlx_chatterbox.model_id or tts_settings.model_id

        return cls(
            profile_id=profile.id,
            profile_name=profile.name,
            audio_driver=speech.audio.driver,
            vad_adapter=speech.vad.adapter,
            vad_threshold=speech.vad.threshold,
            wakeword_model=profile.wakeword.model_path,
            wakeword_threshold=profile.wakeword.threshold,
            stt_adapter=stt_adapter,
            stt_model=stt_model,
            llm_adapter=conversation_adapter,
            llm_model=" · ".join(model_ids),
            tts_adapter=speech.tts.adapter,
            tts_model=tts_model,
        )


@dataclass(frozen=True, slots=True)
class UserMessage:
    turn_id: int
    text: str
    committed: bool = False


@dataclass(frozen=True, slots=True)
class AssistantPhrase:
    phrase_id: int
    text: str
    state: PhraseState = PhraseState.QUEUED


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    reply_id: ReplyId
    phrases: tuple[AssistantPhrase, ...] = ()
    draft: str = ""
    interrupted: bool = False


type ConversationMessage = UserMessage | AssistantMessage


@dataclass(frozen=True, slots=True)
class ConversationState:
    messages: tuple[ConversationMessage, ...] = ()

    def update_transcription(
        self,
        draft: TranscriptionDraft | None,
    ) -> ConversationState:
        if draft is None:
            return self
        messages = list(self.messages)
        for index, message in enumerate(messages):
            if isinstance(message, UserMessage) and message.turn_id == draft.turn_id:
                if not message.committed:
                    messages[index] = replace(message, text=draft.text)
                return replace(self, messages=tuple(messages))
        messages = [
            message
            for message in messages
            if not isinstance(message, UserMessage) or message.committed
        ]
        messages.append(UserMessage(draft.turn_id, draft.text))
        return replace(self, messages=self._bounded(messages))

    def commit_user(self, turn_id: int, text: str) -> ConversationState:
        messages = list(self.messages)
        for index, message in enumerate(messages):
            if isinstance(message, UserMessage) and message.turn_id == turn_id:
                messages[index] = UserMessage(turn_id, text, committed=True)
                return replace(self, messages=tuple(messages))
        messages.append(UserMessage(turn_id, text, committed=True))
        return replace(self, messages=self._bounded(messages))

    def update_reply_draft(self, draft: ReplyDraft | None) -> ConversationState:
        if draft is None:
            return self
        message, index = self._assistant(draft.reply_id)
        if message is None:
            messages = [
                *self.messages,
                AssistantMessage(draft.reply_id, draft=draft.text),
            ]
            return replace(self, messages=self._bounded(messages))
        if message.interrupted:
            return self
        return self._replace_message(index, replace(message, draft=draft.text))

    def start_reply(self, reply_id: ReplyId) -> ConversationState:
        message, _ = self._assistant(reply_id)
        if message is not None:
            return self
        return replace(
            self,
            messages=self._bounded([*self.messages, AssistantMessage(reply_id)]),
        )

    def queue_phrase(
        self,
        reply_id: ReplyId,
        phrase_id: int,
        text: str,
    ) -> ConversationState:
        message, index = self._assistant(reply_id)
        if message is None:
            state = self.start_reply(reply_id)
            message, index = state._assistant(reply_id)
            if message is None:
                return state
            return state.queue_phrase(reply_id, phrase_id, text)
        if message.interrupted or any(
            phrase.phrase_id == phrase_id for phrase in message.phrases
        ):
            return self
        return self._replace_message(
            index,
            replace(
                message,
                phrases=(*message.phrases, AssistantPhrase(phrase_id, text)),
                draft="",
            ),
        )

    def transition_phrase(
        self,
        reply_id: ReplyId,
        phrase_id: int,
        target: PhraseState,
    ) -> ConversationState:
        message, index = self._assistant(reply_id)
        if message is None or message.interrupted:
            return self
        order = {
            PhraseState.QUEUED: 0,
            PhraseState.SPEAKING: 1,
            PhraseState.DELIVERED: 2,
        }
        phrases = tuple(
            replace(phrase, state=target)
            if phrase.phrase_id == phrase_id and order[target] > order[phrase.state]
            else phrase
            for phrase in message.phrases
        )
        return self._replace_message(index, replace(message, phrases=phrases))

    def complete_reply(self, reply_id: ReplyId) -> ConversationState:
        message, index = self._assistant(reply_id)
        if message is None or message.interrupted:
            return self
        return self._replace_message(index, replace(message, draft=""))

    def interrupt_reply(self, reply_id: ReplyId | None) -> ConversationState:
        message, index = (
            self._assistant(reply_id)
            if reply_id is not None
            else self._last_assistant()
        )
        if message is None:
            return self
        return self._replace_message(
            index,
            replace(message, interrupted=True),
        )

    def _assistant(
        self,
        reply_id: ReplyId,
    ) -> tuple[AssistantMessage | None, int]:
        for index, message in enumerate(self.messages):
            if isinstance(message, AssistantMessage) and message.reply_id == reply_id:
                return message, index
        return None, -1

    def _last_assistant(self) -> tuple[AssistantMessage | None, int]:
        for index in range(len(self.messages) - 1, -1, -1):
            message = self.messages[index]
            if isinstance(message, AssistantMessage):
                return message, index
        return None, -1

    def _replace_message(
        self,
        index: int,
        message: ConversationMessage,
    ) -> ConversationState:
        messages = list(self.messages)
        messages[index] = message
        return replace(self, messages=tuple(messages))

    @staticmethod
    def _bounded(
        messages: list[ConversationMessage],
    ) -> tuple[ConversationMessage, ...]:
        return tuple(messages[-200:])


@dataclass(frozen=True, slots=True)
class State:
    info: RuntimeInfo = field(default_factory=RuntimeInfo)
    telemetry: TelemetrySnapshot = field(default_factory=TelemetrySnapshot)
    conversation: ConversationState = field(default_factory=ConversationState)
    conversation_ready: bool = False
    speech_ready: bool = False
    session_mode: VoiceSessionMode | None = None
    generating_replies: frozenset[ReplyId] = frozenset()
    shutting_down: bool = False

    @property
    def is_ready(self) -> bool:
        return self.conversation_ready and self.speech_ready

    @property
    def mode(self) -> RuntimeMode:
        if self.shutting_down:
            return RuntimeMode.SHUTTING_DOWN
        if any(
            isinstance(message, AssistantMessage)
            and any(phrase.state is PhraseState.SPEAKING for phrase in message.phrases)
            for message in self.conversation.messages
        ):
            return RuntimeMode.SPEAKING
        if self.generating_replies:
            return RuntimeMode.THINKING
        if self.telemetry.transcription is not None:
            return RuntimeMode.TRANSCRIBING
        if self.session_mode is VoiceSessionMode.ACTIVE:
            return RuntimeMode.LISTENING
        if self.is_ready:
            return RuntimeMode.WAITING
        return RuntimeMode.STARTING

    def with_runtime(self, profile: Profile, settings: Settings) -> State:
        return replace(self, info=RuntimeInfo.from_runtime(profile, settings))

    def with_telemetry(self, snapshot: TelemetrySnapshot) -> State:
        conversation = self.conversation.update_transcription(snapshot.transcription)
        conversation = conversation.update_reply_draft(snapshot.reply)
        return replace(self, telemetry=snapshot, conversation=conversation)

    def reduce_event(self, event: Event) -> State:
        state = self
        conversation = self.conversation
        match event:
            case ConversationReady():
                state = replace(state, conversation_ready=True)
            case SpeechReady():
                state = replace(state, speech_ready=True)
            case VoiceSessionModeChanged(mode):
                state = replace(state, session_mode=mode)
            case AudioDevicesSelected(driver, devices):
                state = replace(
                    state,
                    info=replace(
                        state.info,
                        audio_driver=driver,
                        input_device=devices.input.name,
                        output_device=devices.output.name,
                    ),
                )
            case UserTurnCommitted(turn_id, text):
                conversation = conversation.commit_user(turn_id, text)
                state = replace(state, conversation=conversation)
            case ReplyGenerationStarted(reply_id):
                conversation = conversation.start_reply(reply_id)
                state = replace(
                    state,
                    conversation=conversation,
                    generating_replies=state.generating_replies | {reply_id},
                )
            case ReplyPhrase(reply_id, phrase_id, text):
                conversation = conversation.queue_phrase(reply_id, phrase_id, text)
                state = replace(state, conversation=conversation)
            case ReplyDraftUpdated(reply_id, text):
                conversation = conversation.update_reply_draft(
                    ReplyDraft(reply_id, text)
                )
                state = replace(state, conversation=conversation)
            case ReplyPhrasePlaybackStarted(reply_id, phrase_id):
                conversation = conversation.transition_phrase(
                    reply_id,
                    phrase_id,
                    PhraseState.SPEAKING,
                )
                state = replace(state, conversation=conversation)
            case ReplyPhraseDelivered(reply_id, phrase_id):
                conversation = conversation.transition_phrase(
                    reply_id,
                    phrase_id,
                    PhraseState.DELIVERED,
                )
                state = replace(state, conversation=conversation)
            case ReplyGenerationCompleted(reply_id):
                conversation = conversation.complete_reply(reply_id)
                state = replace(
                    state,
                    conversation=conversation,
                    generating_replies=state.generating_replies - {reply_id},
                )
            case CancelReply(_, reply_id):
                conversation = conversation.interrupt_reply(reply_id)
                state = replace(
                    state,
                    conversation=conversation,
                    generating_replies=(
                        state.generating_replies - {reply_id}
                        if reply_id is not None
                        else frozenset()
                    ),
                )
            case ShutdownEvent():
                state = replace(state, shutting_down=True)
        return state
