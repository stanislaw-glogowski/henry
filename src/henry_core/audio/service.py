import asyncio
import queue
import threading
from dataclasses import dataclass

from loguru import logger

from ..concurrency import set_future_exception_if_pending, set_future_result_if_pending
from ..events import AppEventSink
from ..lifecycle import AsyncManagedResource
from .domain import AudioFrame
from .events import AudioFrameCaptured, AudioFramePlayed
from .ports import StreamConfig, StreamManager

INPUT_THREAD_NAME = "AudioService.input_worker"
INPUT_STREAM_CONFIG = StreamConfig(
    sample_rate=16_000, channels=1, frames_per_buffer=512
)

OUTPUT_THREAD_NAME = "AudioService.output_worker"
OUTPUT_STREAM_CONFIG = StreamConfig(
    sample_rate=24_000, channels=1, frames_per_buffer=512
)

type InputRequest = ReadStream | None
type OutputRequest = WriteStream | None


@dataclass(frozen=True, slots=True)
class ReadStream:
    frames: asyncio.Queue[AudioFrame]
    response: asyncio.Future[None]


@dataclass(frozen=True, slots=True)
class WriteStream:
    frame: AudioFrame
    response: asyncio.Future[None]


class AudioService(AsyncManagedResource):
    def __init__(
        self,
        streams: StreamManager,
        events: AppEventSink,
    ) -> None:
        self._streams = streams
        self._events = events

        self._input_thread: threading.Thread | None = None
        self._input_cancel = threading.Event()
        self._input_requests: queue.Queue[InputRequest] = queue.Queue(maxsize=1)
        self._output_thread: threading.Thread | None = None
        self._output_requests: queue.Queue[OutputRequest] = queue.Queue()

        self._logger = logger.bind(component="AudioService")

    async def run(
        self,
        input_frames: asyncio.Queue[AudioFrame],
        output_frames: asyncio.Queue[AudioFrame],
    ) -> None:
        """Run audio transfer until cancelled or a worker reports an error."""
        if self._input_thread is None or self._output_thread is None:
            raise RuntimeError("Workers not initialized")

        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(self._input_loop(input_frames))
            tasks.create_task(self._output_loop(output_frames))

    async def open(self) -> None:
        if self._input_thread is not None or self._output_thread is not None:
            raise RuntimeError("Workers already initialized")

        loop = asyncio.get_running_loop()
        input_ready: asyncio.Future[None] = loop.create_future()
        output_ready: asyncio.Future[None] = loop.create_future()

        self._input_cancel.clear()
        self._input_requests = queue.Queue(maxsize=1)
        self._output_requests = queue.Queue()

        self._input_thread = threading.Thread(
            target=self._input_worker,
            args=(loop, input_ready),
            name=INPUT_THREAD_NAME,
        )
        self._output_thread = threading.Thread(
            target=self._output_worker,
            args=(loop, output_ready),
            name=OUTPUT_THREAD_NAME,
        )

        assert self._input_thread is not None and self._output_thread is not None

        try:
            self._input_thread.start()
            await input_ready

            self._output_thread.start()
            await output_ready
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        input_thread = self._input_thread
        output_thread = self._output_thread

        if input_thread is None and output_thread is None:
            return

        self._input_cancel.set()

        try:
            self._input_requests.put_nowait(None)
        except queue.Full:
            pass

        if input_thread is not None:
            await self._join_thread(input_thread)

        self._output_requests.put_nowait(None)

        if output_thread is not None:
            await self._join_thread(output_thread)

        self._input_thread = None
        self._output_thread = None

    async def _input_loop(
        self,
        frames: asyncio.Queue[AudioFrame],
    ) -> None:
        response = asyncio.get_running_loop().create_future()
        self._input_requests.put_nowait(
            ReadStream(
                response=response,
                frames=frames,
            )
        )
        await response

    async def _output_loop(
        self,
        output_frames: asyncio.Queue[AudioFrame],
    ) -> None:
        while True:
            frame = await output_frames.get()
            try:
                response = asyncio.get_running_loop().create_future()

                self._output_requests.put_nowait(
                    WriteStream(
                        frame=frame,
                        response=response,
                    )
                )

                await response
            finally:
                output_frames.task_done()

    def _input_worker(
        self,
        loop: asyncio.AbstractEventLoop,
        ready: asyncio.Future[None],
    ) -> None:
        request: InputRequest = None

        try:
            with self._streams.open_input(INPUT_STREAM_CONFIG) as stream:
                self._logger.trace("Input worker STARTED")

                loop.call_soon_threadsafe(
                    set_future_result_if_pending,
                    ready,
                    None,
                )

                request = self._input_requests.get()

                if request is None:
                    return

                def publish_frame(frame: AudioFrame) -> None:
                    assert request is not None

                    try:
                        request.frames.put_nowait(frame)

                        self._events.publish(
                            AudioFrameCaptured(
                                samples_count=len(frame.samples),
                            )
                        )
                    except asyncio.QueueFull as err:
                        self._input_cancel.set()
                        set_future_exception_if_pending(request.response, err)

                while not self._input_cancel.is_set():
                    loop.call_soon_threadsafe(publish_frame, stream.read())

            self._logger.trace("Input worker STOPPED")

        except BaseException as error:
            self._logger.error("Input worker FAILED: {}", error)

            loop.call_soon_threadsafe(
                set_future_exception_if_pending,
                ready,
                error,
            )

            if request is not None:
                loop.call_soon_threadsafe(
                    set_future_exception_if_pending,
                    request.response,
                    error,
                )
        else:
            if request is not None:
                loop.call_soon_threadsafe(
                    set_future_result_if_pending,
                    request.response,
                    None,
                )

    def _output_worker(
        self,
        loop: asyncio.AbstractEventLoop,
        ready: asyncio.Future[None],
    ) -> None:
        request: OutputRequest = None

        try:
            with self._streams.open_output(OUTPUT_STREAM_CONFIG) as stream:
                self._logger.trace("Output worker STARTED")

                loop.call_soon_threadsafe(
                    set_future_result_if_pending,
                    ready,
                    None,
                )

                while True:
                    request = self._output_requests.get()

                    if request is None:
                        break

                    stream.write(request.frame)

                    self._events.publish(
                        AudioFramePlayed(
                            samples_count=len(request.frame.samples),
                        )
                    )

                    loop.call_soon_threadsafe(
                        set_future_result_if_pending,
                        request.response,
                        None,
                    )

                self._logger.trace("Output worker STOPPED")

        except BaseException as err:
            self._logger.error("Output worker FAILED: {}", err)

            loop.call_soon_threadsafe(
                set_future_exception_if_pending,
                ready,
                err,
            )
            if request is not None:
                loop.call_soon_threadsafe(
                    set_future_exception_if_pending,
                    request.response,
                    err,
                )

    @staticmethod
    async def _join_thread(thread: threading.Thread) -> None:
        if thread.ident is not None:
            await asyncio.to_thread(thread.join)
