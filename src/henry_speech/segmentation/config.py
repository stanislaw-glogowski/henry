from typing import Self

from pydantic import Field, model_validator

from henry_common.validation import ConfigModel


class SegmentationSettings(ConfigModel):
    min_start_speech_frames: int = Field(default=10, gt=0)
    max_start_silence_frames: int = Field(default=150, gt=0)
    max_end_silence_frames: int = Field(default=18, gt=0)
    short_utterance_speech_frames: int = Field(default=31, gt=0)
    short_utterance_end_silence_frames: int = Field(default=28, gt=0)
    max_utterance_frames: int = Field(default=1_875, gt=0)
    pre_roll_frames: int = Field(default=15, ge=0)

    @model_validator(mode="after")
    def validate_adaptive_limits(self) -> Self:
        if self.short_utterance_end_silence_frames < self.max_end_silence_frames:
            raise ValueError(
                "short_utterance_end_silence_frames must be greater than or equal "
                "to max_end_silence_frames"
            )
        if self.max_utterance_frames <= self.short_utterance_speech_frames:
            raise ValueError(
                "max_utterance_frames must exceed short_utterance_speech_frames"
            )
        return self
