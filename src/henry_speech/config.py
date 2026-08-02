from henry_common.validation import ConfigModel

from .audio import AudioSettings
from .capture import VADSettings, WakeWordProfile, WakeWordSettings
from .segmentation import SegmentationSettings
from .synthesis import TTSProfile, TTSSettings, default_tts_settings
from .transcription import STTProfile, STTSettings, default_stt_settings


class SpeechSettings(ConfigModel):
    audio: AudioSettings = AudioSettings()
    vad: VADSettings = VADSettings()
    wakeword: WakeWordSettings = WakeWordSettings()
    segmentation: SegmentationSettings = SegmentationSettings()
    tts: TTSSettings = default_tts_settings()
    stt: STTSettings = default_stt_settings()


class SpeechProfile(TTSProfile, STTProfile):
    wakeword: WakeWordProfile
