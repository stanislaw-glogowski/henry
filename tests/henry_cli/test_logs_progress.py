from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import RLock

import pytest
from loguru import logger

from henry_cli.logs import LogBuffer, configure_ui_logger
from henry_cli.progress import (
    HuggingFaceProgress,
    ProgressStatus,
    ProgressStore,
    _ProgressTqdm,
)


def test_log_buffer_is_bounded_thread_safe_sink() -> None:
    with pytest.raises(ValueError, match="positive"):
        LogBuffer(0)
    buffer = LogBuffer(2)
    buffer.write("one\n")
    buffer.write("  ")
    buffer.write("two")
    buffer.write("three")
    assert buffer.drain() == ("two", "three")
    assert buffer.drain() == ()

    configure_ui_logger(buffer, "INFO")
    logger.info("ready")
    logger.debug("hidden")
    assert any("ready" in line and "Henry" in line for line in buffer.drain())


def test_progress_store_tracks_updates_completion_and_pruning() -> None:
    store = ProgressStore()
    item_id = store.begin("model.bin", 0, 100, "B")
    assert store.snapshot.is_active
    assert store.snapshot.items[0].percentage == 0
    store.update(item_id, completed=25, total=200)
    assert store.snapshot.items[0].total == 100
    store.update(
        item_id,
        description="weights.bin",
        completed=150,
        total=200,
        replace_total=True,
    )
    assert store.snapshot.items[0].percentage == 75
    store.complete(item_id, 240, 200)
    item = store.snapshot.items[0]
    assert item.percentage == 100
    assert item.status is ProgressStatus.COMPLETED
    assert not store.snapshot.is_active
    store.update(999, completed=1)
    store.complete(999, 1, 1)
    store.clear_completed()
    assert store.snapshot.items == ()

    for index in range(store._MAX_COMPLETED + 3):
        current = store.begin(str(index), 0, None, "files")
        store.complete(current, 1, None)
    store.begin("active", 0, 0, "items")
    assert len(store.snapshot.items) <= store._MAX_COMPLETED + 1
    assert store.snapshot.items[-1].percentage is None


def test_custom_hugging_face_progress_behaves_like_tqdm() -> None:
    store = ProgressStore()
    _ProgressTqdm.store = store
    bar = _ProgressTqdm(desc="weights", total=4, initial=1, unit="B")
    assert bar.total == 4
    assert bar.format_dict["rate"] is not None
    bar.update()
    bar.update_transfer(1)
    bar.total = 6
    bar.set_description("model")
    bar.set_description_str("model.bin", refresh=False)
    bar.set_postfix_str("fast")
    bar.set_transfer_postfix_str("fast", refresh=False)
    bar.refresh()
    bar.close()
    bar.close()
    assert store.snapshot.items[0].status is ProgressStatus.COMPLETED
    assert store.snapshot.items[0].description == "model.bin"

    with _ProgressTqdm([1, 2], desc="files", total=2) as iterable:
        assert list(iterable) == [1, 2]
    assert store.snapshot.items[-1].completed == 2

    empty = _ProgressTqdm(desc="empty")
    assert list(empty) == []
    empty.close()
    _ProgressTqdm.store = None
    with pytest.raises(RuntimeError, match="not configured"):
        _ProgressTqdm()


def test_hugging_face_progress_injects_and_restores_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib import import_module

    import huggingface_hub
    from tqdm import tqdm

    snapshot_module = import_module("huggingface_hub._snapshot_download")

    calls: list[tuple[str, object]] = []

    def download(name: str, **kwargs):
        calls.append((name, kwargs.get("tqdm_class")))
        return name

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(snapshot_module, "hf_hub_download", download)
    original_snapshot = huggingface_hub.snapshot_download
    original_file = huggingface_hub.hf_hub_download
    original_internal_file = snapshot_module.hf_hub_download
    adapter = HuggingFaceProgress(ProgressStore())
    original_tqdm_lock = getattr(tqdm, "_lock", adapter._MISSING)
    with adapter:
        assert huggingface_hub.snapshot_download("snapshot") == "snapshot"
        assert huggingface_hub.hf_hub_download("file", tqdm_class=str) == "file"
        assert snapshot_module.hf_hub_download("internal", tqdm_class=str) == "internal"
        with pytest.raises(RuntimeError, match="already installed"):
            adapter.__enter__()
    assert calls == [
        ("snapshot", _ProgressTqdm),
        ("file", str),
        ("internal", _ProgressTqdm),
    ]
    assert huggingface_hub.snapshot_download is original_snapshot
    assert huggingface_hub.hf_hub_download is original_file
    assert snapshot_module.hf_hub_download is original_internal_file
    if original_tqdm_lock is adapter._MISSING:
        assert not hasattr(tqdm, "_lock")
    else:
        assert tqdm.get_lock() is original_tqdm_lock
    assert _ProgressTqdm.store is None

    adapter.__exit__(None, None, None)


def test_hugging_face_progress_prevents_tqdm_multiprocessing_lock() -> None:
    from tqdm import tqdm

    original_lock = getattr(tqdm, "_lock", HuggingFaceProgress._MISSING)
    if original_lock is not HuggingFaceProgress._MISSING:
        del tqdm._lock
    try:
        with HuggingFaceProgress(ProgressStore()):
            assert isinstance(tqdm.get_lock(), type(RLock()))
            with ThreadPoolExecutor(max_workers=1) as executor:
                result = executor.submit(
                    lambda: list(tqdm(range(1), disable=True))
                ).result()
            assert result == [0]
        assert not hasattr(tqdm, "_lock")
    finally:
        if original_lock is not HuggingFaceProgress._MISSING:
            tqdm.set_lock(original_lock)
