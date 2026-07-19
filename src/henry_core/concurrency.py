import asyncio


def put_latest[T](
    queue: asyncio.Queue[T],
    value: T,
) -> None:
    if queue.full():
        try:
            queue.get_nowait()
            queue.task_done()
        except asyncio.QueueEmpty:
            pass

    queue.put_nowait(value)


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
