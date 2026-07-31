from henry_common.validation import ConfigModel


class SegmentationSettings(ConfigModel):
    min_start_speech_frames: int = 10
    max_start_silence_frames: int = 150
    max_end_silence_frames: int = 50
    pre_roll_frames: int = 15
