import asyncio
from abc import ABC, abstractmethod
from asyncio import Future
from collections.abc import Callable
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from types import TracebackType
from typing import Self, TypeVar, TypeVarTuple, Unpack

from .logger import bind_logger

_T = TypeVar("_T")
_Ts = TypeVarTuple("_Ts")


class Component:
    def __init__(self, context: str | None = None):
        self._logger = bind_logger(self, context)


class AbstractResource(AbstractContextManager, Component, ABC):
    def __init__(self, context: str | None = None) -> None:
        super().__init__(context)

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None:
        self.close()

    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class AbstractService(Component):
    def __init__(self) -> None:
        super().__init__()


class AbstractAsyncService(AbstractService, AbstractAsyncContextManager, ABC):
    def __init__(self) -> None:
        super().__init__()
        self._executor: ThreadPoolExecutor | None = None

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None:
        await self.stop()

    async def start(self) -> None:
        if self._executor is not None:
            raise RuntimeError(f"{self.__class__.__name__} is already started")

        loop = asyncio.get_running_loop()
        try:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"{self.__class__.__name__}_executor",
            )
            await loop.run_in_executor(
                self._executor,
                self._open_resources,
            )
            self._logger.debug("Service STARTED")
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        if self._executor is None:
            return
        loop = asyncio.get_running_loop()

        errors: list[Exception] = []

        try:
            await loop.run_in_executor(self._executor, self._close_resources)
        except Exception as err:
            errors.append(err)

        try:
            await asyncio.to_thread(self._executor.shutdown)
        except Exception as err:
            errors.append(err)
        finally:
            self._executor = None

        try:
            await self._post_stop()
        except Exception as err:
            errors.append(err)
        finally:
            if errors:
                self._logger.warning("Service STOPPED", errors=errors)
            else:
                self._logger.debug("Service STOPPED")

    async def _post_stop(self) -> None:
        pass

    @abstractmethod
    def _open_resources(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _close_resources(self) -> None:
        raise NotImplementedError

    def _require_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            raise RuntimeError(f"{self.__class__.__name__} is not started")

        return self._executor

    def _run_in_executor(
        self, func: Callable[[*Unpack[_Ts]], _T], *args: *_Ts
    ) -> Future[_T]:
        executor = self._require_executor()
        loop = asyncio.get_running_loop()

        return loop.run_in_executor(executor, func, *args)
