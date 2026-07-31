from henry_common.validation import ConfigModel

from .audio import AudioSettings
from .capture import VADSettings, WakeWordProfile, WakeWordSettings
from .segmentation import SegmentationSettings
from .synthesis import TTSProfile, TTSSettings
from .transcription import STTProfile, STTSettings


class SpeechSettings(ConfigModel):
    audio: AudioSettings = AudioSettings()
    vad: VADSettings = VADSettings()
    wakeword: WakeWordSettings = WakeWordSettings()
    segmentation: SegmentationSettings = SegmentationSettings()
    tts: TTSSettings = TTSSettings()
    stt: STTSettings = STTSettings()


class SpeechProfile(ConfigModel):
    wakeword: WakeWordProfile
    tts: TTSProfile
    stt: STTProfile = STTProfile()
