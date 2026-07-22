import asyncio
import threading


async def join_started_thread(thread: threading.Thread) -> None:
    if thread.ident is not None:
        await asyncio.to_thread(thread.join)


def set_future_exception_if_pending[T](
    future: asyncio.Future[T],
    error: BaseException,
) -> None:
    if not future.done():
        future.set_exception(error)


def set_future_result_if_pending[T](
    future: asyncio.Future[T],
    result: T,
) -> None:
    if not future.done():
        future.set_result(result)
