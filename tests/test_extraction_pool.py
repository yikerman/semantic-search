import os
import time

import pytest

from semsearch.cli.crawl.extraction import ExtractionPool


def slow_task(marker: str) -> None:
    from pathlib import Path

    Path(marker).write_text(str(os.getpid()))
    time.sleep(30)


def identify_worker() -> int:
    return os.getpid()


async def test_timeout_kills_worker_and_next_task_succeeds(tmp_path):
    pool = ExtractionPool(workers=1, timeout=2)
    marker = tmp_path / "worker"
    try:
        with pytest.raises(TimeoutError):
            await pool.call(slow_task, str(marker))
        old_pid = int(marker.read_text())
        new_pid = await pool.call(identify_worker)
        assert new_pid != old_pid
        with pytest.raises(ProcessLookupError):
            os.kill(old_pid, 0)
    finally:
        await pool.close()
