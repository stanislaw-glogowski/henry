import asyncio
import threading
from collections.abc import Iterator

import numpy as np
import pytest

from henry_speech.audio import (
    AudioFormat,
    AudioFrame,
    AudioInput,
    AudioOutput,
    AudioPlaybackOutcome,
)
from henry_speech.capture import (
    CaptureService,
    DetectionResult,
    VADModel,
    WakeWordModel,
)
from henry_speech.playback import PlaybackService
from henry_speech.synthesis import SynthesisService, TTSModel
from henry_speech.transcription import (
    STTModel,
    TranscriptionChunk,
    TranscriptionService,
    TranscriptionText,
)

FORMAT = AudioFormat(sample_rate=16_000, channels=1)


def audio(value: float = 0.1) -> AudioFrame:
    return FORMAT.build_frame(np.asarray([value], dtype=np.float32))


class FakeAudioInput(AudioInput):
    def __init__(self, frames: list[AudioFrame]) -> None:
        super().__init__()
        self.frames = iter(frames)
        self.calls: list[str] = []

    def read(self) -> AudioFrame:
        try:
            return next(self.frames)
        except StopIteration as error:
            raise EOFError("done") from error

    def open(self) -> None:
        self.calls.append("open")

    def close(self) -> None:
        self.calls.append("close")


class FakeVAD(VADModel):
    def __init__(self, result: DetectionResult | None = None) -> None:
        super().__init__()
        self.result = result or DetectionResult(score=1.0, detected=True)
        self.calls: list[str] = []

    def detect(self, frame: AudioFrame) -> DetectionResult:
        return self.result

    def open(self) -> None:
        self.calls.append("open")

    def close(self) -> None:
        self.calls.append("close")


class FakeWakeWord(WakeWordModel):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def detect(self, frame: AudioFrame) -> DetectionResult:
        return DetectionResult(score=0.9, detected=True)

    def reset(self) -> None:
        self.calls.append("reset")

    def open(self) -> None:
        self.calls.append("open")

    def close(self) -> None:
        self.calls.append("close")


class FakeSTT(STTModel):
    def __init__(
        self,
        chunks: list[TranscriptionChunk],
        error: BaseException | None = None,
    ) -> None:
        super().__init__()
        self.chunks = chunks
        self.error = error
        self.calls: list[str] = []
        self.thread_ids: set[int] = set()

    def transcribe(self, frame: AudioFrame) -> Iterator[TranscriptionChunk]:
        self.thread_ids.add(threading.get_ident())
        if self.error:
            raise self.error
        yield from self.chunks

    def open(self) -> None:
        self.calls.append("open")
        self.thread_ids.add(threading.get_ident())

    def close(self) -> None:
        self.calls.append("close")
        self.thread_ids.add(threading.get_ident())


class FakeTTS(TTSModel):
    def __init__(
        self,
        frames: list[AudioFrame],
        error: BaseException | None = None,
    ) -> None:
        super().__init__()
        self.frames = frames
        self.error = error
        self.calls: list[str] = []

    def synthesize(self, text: str) -> Iterator[AudioFrame]:
        if self.error:
            raise self.error
        yield from self.frames

    def open(self) -> None:
        self.calls.append("open")

    def close(self) -> None:
        self.calls.append("close")


class FakeAudioOutput(AudioOutput):
    def __init__(self) -> None:
        super().__init__()
        self.frames: list[AudioFrame] = []
        self.calls: list[str] = []

    def write(self, frame: AudioFrame) -> AudioPlaybackOutcome:
        self.frames.append(frame)
        return AudioPlaybackOutcome.PLAYED

    def interrupt(self) -> None:
        self.calls.append("interrupt")

    def duck(self) -> None:
        self.calls.append("duck")

    def restore(self) -> None:
        self.calls.append("restore")

    def open(self) -> None:
        self.calls.append("open")

    def close(self) -> None:
        self.calls.append("close")


def test_capture_service_lifecycle_detection_and_errors() -> None:
    async def scenario() -> None:
        source = FakeAudioInput([audio()])
        vad = FakeVAD(DetectionResult(score=0.8, detected=True))
        wakeword = FakeWakeWord()
        service = CaptureService(source, vad, wakeword)

        async with service:
            service.enable_wakeword()
            service.enable_wakeword()
            stream = service.capture()
            captured = await asyncio.wait_for(stream.__anext__(), 1)
            assert captured.vad.detected
            assert captured.wakeword.detected
            assert "reset" in wakeword.calls
            with pytest.raises(RuntimeError, match="already in progress"):
                await service.capture().__anext__()
            await stream.aclose()
            service.disable_wakeword()
            service.disable_wakeword()

        assert source.calls == ["open", "close"]
        assert vad.calls == ["open", "close"]
        assert wakeword.calls[0] == "open"
        assert wakeword.calls[-1] == "close"

        failing = CaptureService(FakeAudioInput([]), FakeVAD(), FakeWakeWord())
        async with failing:
            with pytest.raises(EOFError, match="done"):
                await failing.capture().__anext__()

    asyncio.run(scenario())


def test_transcription_service_outputs_text_empty_and_errors() -> None:
    async def collect(service: TranscriptionService):
        async with service:
            return [item async for item in service.transcribe(audio())]

    model = FakeSTT(
        [TranscriptionChunk("Hello"), TranscriptionChunk(""), TranscriptionChunk("!")]
    )
    assert asyncio.run(collect(TranscriptionService(model))) == [
        TranscriptionChunk("Hello"),
        TranscriptionChunk("!"),
        TranscriptionText("Hello!"),
    ]
    assert len(model.thread_ids) == 1
    assert asyncio.run(collect(TranscriptionService(FakeSTT([])))) == [None]

    async def error_scenario() -> None:
        service = TranscriptionService(FakeSTT([], RuntimeError("stt")))
        async with service:
            with pytest.raises(RuntimeError, match="stt"):
                await service.transcribe(audio()).__anext__()

    asyncio.run(error_scenario())


def test_synthesis_and_playback_services() -> None:
    async def scenario() -> None:
        frame = audio()
        model = FakeTTS([frame])
        synthesis = SynthesisService(model)
        async with synthesis:
            assert [item async for item in synthesis.synthesize("hello")] == [frame]
        assert model.calls == ["open", "close"]

        failing = SynthesisService(FakeTTS([], RuntimeError("tts")))
        async with failing:
            with pytest.raises(RuntimeError, match="tts"):
                await failing.synthesize("hello").__anext__()

        output = FakeAudioOutput()
        playback = PlaybackService(output)
        async with playback:
            assert await playback.play(frame) is AudioPlaybackOutcome.PLAYED
            await playback.duck()
            await playback.restore()
            await playback.interrupt()
        assert output.frames == [frame]
        assert output.calls == ["open", "duck", "restore", "interrupt", "close"]

    asyncio.run(scenario())
