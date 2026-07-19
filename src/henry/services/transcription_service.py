import asyncio

from loguru import logger

from ..domain import SpeechSegment, SpeechTranscription
from ..events import PipelineStage, PipelineStageChanged, PipelineStageStatus
from ..ports import AppEventSink, SpeechTranscriber, TelemetrySink


class TranscriptionService:
    def __init__(self, transcriber: SpeechTranscriber) -> None:
        self._transcriber = transcriber
        self._logger = logger.bind(component="TranscriptionService")

    async def run(
        self,
        segments: asyncio.Queue[SpeechSegment],
        transcriptions: asyncio.Queue[SpeechTranscription],
        recording: asyncio.Event,
        events: AppEventSink,
        telemetry: TelemetrySink,
    ) -> None:
        self._logger.debug("Running")

        events.publish(
            PipelineStageChanged(
                PipelineStage.TRANSCRIPTION, PipelineStageStatus.READY
            ),
        )

        while True:
            segment = await segments.get()

            try:
                self._logger.trace(
                    "Sending request, audio: [{}]", len(segment.audio.samples)
                )

                events.publish(
                    PipelineStageChanged(
                        PipelineStage.TRANSCRIPTION, PipelineStageStatus.STARTED
                    ),
                )

                transcription = await self._transcriber.transcribe(segment)

                if transcription is None:
                    self._logger.trace("Request COMPLETED: no text")

                    recording.set()
                    self._logger.debug("Recording ENABLED")

                    events.publish(
                        PipelineStageChanged(
                            PipelineStage.TRANSCRIPTION, PipelineStageStatus.COMPLETED
                        ),
                        PipelineStageChanged(
                            PipelineStage.RECORDING, PipelineStageStatus.STARTED
                        ),
                    )
                    continue

                self._logger.trace("Request COMPLETED: text='{}'", transcription.text)

                events.publish(
                    PipelineStageChanged(
                        PipelineStage.TRANSCRIPTION, PipelineStageStatus.COMPLETED
                    ),
                )

                await transcriptions.put(transcription)

            finally:
                segments.task_done()
