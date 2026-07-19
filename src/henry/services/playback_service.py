import asyncio

from loguru import logger

from ..domain import AudioFrame
from ..events import PipelineStage, PipelineStageChanged, PipelineStageStatus
from ..ports import AppEventSink, AudioOutput, TelemetrySink
from ..telemetry import AudioFramePlayed


class PlaybackService:
    def __init__(self, audio: AudioOutput) -> None:
        self._audio = audio
        self._logger = logger.bind(component="PlaybackService")

    async def run(
        self,
        frames: asyncio.Queue[AudioFrame | None],
        recording: asyncio.Event,
        events: AppEventSink,
        telemetry: TelemetrySink,
    ) -> None:
        self._logger.debug("Running")

        events.publish(
            PipelineStageChanged(PipelineStage.PLAYBACK, PipelineStageStatus.READY)
        )

        while True:
            chunk = await frames.get()

            try:
                if chunk is None:
                    recording.set()

                    self._logger.debug("Recording ENABLED")

                    events.publish(
                        PipelineStageChanged(
                            PipelineStage.PLAYBACK, PipelineStageStatus.COMPLETED
                        ),
                        PipelineStageChanged(
                            PipelineStage.RECORDING, PipelineStageStatus.STARTED
                        ),
                    )
                    continue

                await self._audio.play(chunk)

                telemetry.publish(AudioFramePlayed(len(chunk.samples)))
            finally:
                frames.task_done()
