from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum, auto
from functools import wraps
from threading import Lock, RLock
from time import monotonic
from types import TracebackType
from typing import Any, Self


class ProgressStatus(Enum):
    ACTIVE = auto()
    COMPLETED = auto()


@dataclass(frozen=True, slots=True)
class ProgressItem:
    id: int
    description: str
    completed: float
    total: float | None
    unit: str
    status: ProgressStatus

    @property
    def percentage(self) -> float | None:
        if self.total is None or self.total <= 0:
            return None
        return min(100.0, max(0.0, self.completed / self.total * 100.0))


@dataclass(frozen=True, slots=True)
class ProgressSnapshot:
    items: tuple[ProgressItem, ...] = ()

    @property
    def is_active(self) -> bool:
        return any(item.status is ProgressStatus.ACTIVE for item in self.items)


class ProgressStore:
    _MAX_COMPLETED = 24

    def __init__(self) -> None:
        self._items: dict[int, ProgressItem] = {}
        self._lock = Lock()
        self._sequence = 0

    @property
    def snapshot(self) -> ProgressSnapshot:
        with self._lock:
            return ProgressSnapshot(tuple(self._items.values()))

    def begin(
        self,
        description: str,
        completed: float,
        total: float | None,
        unit: str,
    ) -> int:
        with self._lock:
            self._prune_completed()
            self._sequence += 1
            item_id = self._sequence
            self._items[item_id] = ProgressItem(
                id=item_id,
                description=description,
                completed=completed,
                total=total,
                unit=unit,
                status=ProgressStatus.ACTIVE,
            )
            return item_id

    def update(
        self,
        item_id: int,
        *,
        description: str | None = None,
        completed: float | None = None,
        total: float | None = None,
        replace_total: bool = False,
    ) -> None:
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                return
            self._items[item_id] = ProgressItem(
                id=item.id,
                description=(
                    description if description is not None else item.description
                ),
                completed=(completed if completed is not None else item.completed),
                total=(total if replace_total else item.total),
                unit=item.unit,
                status=item.status,
            )

    def complete(self, item_id: int, completed: float, total: float | None) -> None:
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                return
            self._items[item_id] = ProgressItem(
                id=item.id,
                description=item.description,
                completed=completed,
                total=total,
                unit=item.unit,
                status=ProgressStatus.COMPLETED,
            )

    def clear_completed(self) -> None:
        with self._lock:
            self._items = {
                item_id: item
                for item_id, item in self._items.items()
                if item.status is ProgressStatus.ACTIVE
            }

    def _prune_completed(self) -> None:
        completed = [
            item_id
            for item_id, item in self._items.items()
            if item.status is ProgressStatus.COMPLETED
        ]
        for item_id in completed[: -self._MAX_COMPLETED]:
            del self._items[item_id]


class _ProgressTqdm:
    store: ProgressStore | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        store = self.store
        if store is None:
            raise RuntimeError("Hugging Face progress store is not configured")
        self._iterable: Iterable[Any] | None = args[0] if args else None
        self.desc = str(kwargs.get("desc") or "Downloading")
        self.unit = str(kwargs.get("unit") or "items")
        self.n = float(kwargs.get("initial") or 0)
        self._total = self._normalize_total(kwargs.get("total"))
        self._started_at = monotonic()
        self._closed = False
        self._item_id = store.begin(self.desc, self.n, self._total, self.unit)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None:
        self.close()

    def __iter__(self) -> Iterator[Any]:
        if self._iterable is None:
            return
        for item in self._iterable:
            yield item
            self.update()

    @property
    def total(self) -> float | None:
        return self._total

    @total.setter
    def total(self, value: float | int | None) -> None:
        self._total = self._normalize_total(value)
        self._update_store(replace_total=True)

    @property
    def format_dict(self) -> dict[str, float | None]:
        elapsed = monotonic() - self._started_at
        return {"rate": self.n / elapsed if elapsed > 0 else None}

    def update(self, amount: float | int | None = 1) -> None:
        self.n += float(amount or 0)
        self._update_store()

    def update_transfer(self, amount: float | int | None = 1) -> None:
        self.update(amount)

    def refresh(self) -> None:
        self._update_store(replace_total=True)

    def set_description(self, description: str, refresh: bool = True) -> None:
        self.desc = description
        self._update_store(description=description)
        if refresh:
            self.refresh()

    def set_description_str(self, description: str, refresh: bool = True) -> None:
        self.set_description(description, refresh)

    def set_postfix_str(self, _postfix: str, refresh: bool = True) -> None:
        if refresh:
            self.refresh()

    def set_transfer_postfix_str(
        self,
        postfix: str,
        refresh: bool = True,
    ) -> None:
        self.set_postfix_str(postfix, refresh)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        store = self.store
        if store is not None:
            store.complete(self._item_id, self.n, self._total)

    def _update_store(
        self,
        *,
        description: str | None = None,
        replace_total: bool = False,
    ) -> None:
        store = self.store
        if store is not None:
            store.update(
                self._item_id,
                description=description,
                completed=self.n,
                total=self._total,
                replace_total=replace_total,
            )

    @staticmethod
    def _normalize_total(value: Any) -> float | None:
        if value is None:
            return None
        total = float(value)
        return total if total > 0 else None


class HuggingFaceProgress(AbstractContextManager):
    _MISSING = object()

    def __init__(self, store: ProgressStore) -> None:
        self._store = store
        self._original_snapshot_download: Any = None
        self._original_hf_hub_download: Any = None
        self._original_internal_hf_hub_download: Any = None
        self._original_tqdm_lock: Any = self._MISSING

    def __enter__(self) -> Self:
        if self._original_snapshot_download is not None:
            raise RuntimeError("Hugging Face progress adapter is already installed")

        import huggingface_hub
        from huggingface_hub import _snapshot_download as snapshot_module
        from tqdm import tqdm

        self._original_tqdm_lock = getattr(tqdm, "_lock", self._MISSING)
        tqdm.set_lock(RLock())
        _ProgressTqdm.store = self._store
        self._original_snapshot_download = huggingface_hub.snapshot_download
        self._original_hf_hub_download = huggingface_hub.hf_hub_download
        self._original_internal_hf_hub_download = snapshot_module.hf_hub_download

        huggingface_hub.snapshot_download = self._with_progress(
            self._original_snapshot_download
        )
        huggingface_hub.hf_hub_download = self._with_progress(
            self._original_hf_hub_download
        )
        snapshot_module.hf_hub_download = self._with_progress(
            self._original_internal_hf_hub_download,
            force=True,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None:
        import huggingface_hub
        from huggingface_hub import _snapshot_download as snapshot_module
        from tqdm import tqdm

        if self._original_snapshot_download is not None:
            huggingface_hub.snapshot_download = self._original_snapshot_download
        if self._original_hf_hub_download is not None:
            huggingface_hub.hf_hub_download = self._original_hf_hub_download
        if self._original_internal_hf_hub_download is not None:
            snapshot_module.hf_hub_download = self._original_internal_hf_hub_download
        self._original_snapshot_download = None
        self._original_hf_hub_download = None
        self._original_internal_hf_hub_download = None
        if self._original_tqdm_lock is self._MISSING:
            if hasattr(tqdm, "_lock"):
                del tqdm._lock
        else:
            tqdm.set_lock(self._original_tqdm_lock)
        self._original_tqdm_lock = self._MISSING
        _ProgressTqdm.store = None

    @staticmethod
    def _with_progress(function: Any, *, force: bool = False) -> Any:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if force:
                kwargs["tqdm_class"] = _ProgressTqdm
            else:
                kwargs.setdefault("tqdm_class", _ProgressTqdm)
            return function(*args, **kwargs)

        return wrapped
