import asyncio

from loguru import logger

from ..concurrency import put_latest
from ..domain import AudioFrame
from ..events import PipelineStage, PipelineStageChanged, PipelineStageStatus
from ..ports import AppEventSink, AudioInput, TelemetrySink
from ..telemetry import AudioFrameCaptured


class CaptureService:
    def __init__(self, audio: AudioInput) -> None:
        self._audio = audio
        self._logger = logger.bind(component="CaptureService")

    async def run(
        self,
        frames: asyncio.Queue[AudioFrame],
        events: AppEventSink,
        telemetry: TelemetrySink,
        shutdown: asyncio.Event,
    ) -> None:
        self._logger.debug("Running")

        events.publish(
            PipelineStageChanged(PipelineStage.CAPTURE, PipelineStageStatus.READY),
            PipelineStageChanged(PipelineStage.CAPTURE, PipelineStageStatus.STARTED),
        )

        while not shutdown.is_set():
            frame = await self._audio.read()
            put_latest(frames, frame)

            telemetry.publish(AudioFrameCaptured(len(frame.samples)))

        events.publish(
            PipelineStageChanged(PipelineStage.CAPTURE, PipelineStageStatus.COMPLETED),
        )
