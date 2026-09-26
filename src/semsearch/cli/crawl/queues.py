from pathlib import Path
from typing import BinaryIO, Literal

from scrapy.squeues import PickleFifoDiskQueue


# Scrapy generates this public queue class through serialization wrappers.
class UnbufferedFifoDiskQueue(PickleFifoDiskQueue):  # ty: ignore[unsupported-base]
    """Keep Scrapy serialization and FIFO behavior without unused file buffers."""

    path: str

    def _openchunk(self, number: int, mode: Literal["rb", "ab+"] = "rb") -> BinaryIO:
        # queuelib accesses these descriptors with os.read/write/lseek. Buffered
        # wrappers waste two 128 KiB buffers per queue on Python 3.14.
        return Path(self.path, f"q{number:05d}").open(mode, buffering=0)
