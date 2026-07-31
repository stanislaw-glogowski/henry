from typing import Literal

from henry_common.validation import ConfigModel


class AudioSettings(ConfigModel):
    driver: Literal["pyaudio"] = "pyaudio"
