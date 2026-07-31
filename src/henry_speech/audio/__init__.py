from .config import AudioSettings
from .domain import (
    AudioBuffer,
    AudioFormat,
    AudioFrame,
    AudioSamples,
)
from .ports import (
    AudioDriver,
    AudioInput,
    AudioOutput,
)


def get_audio_driver(settings: AudioSettings) -> AudioDriver:
    from .adapters import get_audio_driver as create_audio_driver

    return create_audio_driver(settings)


__all__ = [
    "AudioBuffer",
    "AudioDriver",
    "AudioFormat",
    "AudioFrame",
    "AudioInput",
    "AudioOutput",
    "AudioSamples",
    "AudioSettings",
    "get_audio_driver",
]
