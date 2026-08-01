from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import AudioSettings
    from ..ports import AudioDriver


def get_audio_driver(settings: AudioSettings) -> AudioDriver:
    match settings.driver:
        case "avfaudio":
            from .avfaudio import AVFAudioDriver

            return AVFAudioDriver()
        case "pyaudio":
            from .pyaudio import PyAudioDriver

            return PyAudioDriver()
        case _:
            raise ValueError(f"Unsupported audio driver: {settings.driver!r}")
