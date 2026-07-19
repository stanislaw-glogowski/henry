import asyncio

from loguru import logger

from ..domain import AssistantReply, AudioFrame
from ..events import PipelineStage, PipelineStageChanged, PipelineStageStatus
from ..ports import AppEventSink, SpeechSynthesizer, TelemetrySink


class SynthesiseA:
    pass


class SynthesiseService:
    def __init__(self, synthesizer: SpeechSynthesizer) -> None:
        self._synthesizer = synthesizer
        self._logger = logger.bind(component="SynthesiseService")

    async def run(
        self,
        replies: asyncio.Queue[AssistantReply],
        frames: asyncio.Queue[AudioFrame | None],
        events: AppEventSink,
        telemetry: TelemetrySink,
    ) -> None:
        self._logger.debug("Running")

        events.publish(
            PipelineStageChanged(PipelineStage.SYNTHESIS, PipelineStageStatus.READY),
        )

        while True:
            reply = await replies.get()

            try:
                self._logger.trace("Sending request: text='{}'", reply.text)

                events.publish(
                    PipelineStageChanged(
                        PipelineStage.SYNTHESIS, PipelineStageStatus.STARTED
                    ),
                )

                total_frames = 0

                async for frame in self._synthesizer.synthesize(reply.text):
                    total_frames += 1
                    self._logger.trace("Response chunk: frame=[{}]", len(frame.samples))
                    await frames.put(frame)

                await frames.put(None)

                self._logger.trace("Request COMPLETED: total_frames={}", total_frames)

                events.publish(
                    PipelineStageChanged(
                        PipelineStage.SYNTHESIS, PipelineStageStatus.COMPLETED
                    ),
                )

            finally:
                replies.task_done()
