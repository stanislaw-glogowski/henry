import asyncio

from loguru import logger

from ..domain import AudioFrame, SpeechSegment, SpeechSegmenter
from ..events import PipelineStage, PipelineStageChanged, PipelineStageStatus
from ..ports import AppEventSink, TelemetrySink, VoiceActivityDetector
from ..telemetry import VadObserved


class RecordingService:
    def __init__(
        self,
        vad: VoiceActivityDetector,
        segmenter: SpeechSegmenter | None = None,
    ) -> None:
        if segmenter is None:
            segmenter = SpeechSegmenter()

        assert isinstance(segmenter, SpeechSegmenter)

        self._vad = vad
        self._segmenter = segmenter
        self._logger = logger.bind(component="RecordingService")

    async def run(
        self,
        frames: asyncio.Queue[AudioFrame],
        segments: asyncio.Queue[SpeechSegment],
        recording: asyncio.Event,
        events: AppEventSink,
        telemetry: TelemetrySink,
    ) -> None:
        self._logger.debug("Running")
        self._logger.debug("Recording ENABLED")

        events.publish(
            PipelineStageChanged(PipelineStage.RECORDING, PipelineStageStatus.READY),
            PipelineStageChanged(PipelineStage.RECORDING, PipelineStageStatus.STARTED),
        )

        while True:
            frame = await frames.get()

            try:
                frame = self._vad.analyze(frame)

                telemetry.publish(
                    VadObserved(
                        score=frame.speech_score,
                        is_speech=frame.is_speech,
                    )
                )

                if not recording.is_set():
                    continue

                ready, segment = self._segmenter.feed(frame)

                if not ready:
                    continue

                if segment is None:
                    self._logger.trace("Segment SKIPPED: no audio")
                    continue

                self._logger.trace(
                    "Segment BUILT: audio=[{}]",
                    len(segment.audio.samples),
                )

                recording.clear()

                self._logger.debug("Recording DISABLED")

                events.publish(
                    PipelineStageChanged(
                        PipelineStage.RECORDING, PipelineStageStatus.COMPLETED
                    ),
                    PipelineStageChanged(
                        PipelineStage.TRANSCRIPTION, PipelineStageStatus.STARTED
                    ),
                )

                await segments.put(segment)
            finally:
                frames.task_done()
