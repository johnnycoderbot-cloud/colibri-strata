"""Bounded buffered file-tier reads for NVMe/SSD expert/head artifacts.

Uses ordinary files, never raw devices. This is not O_DIRECT or GPUDirect.
Each future retains immutable request identity and verifies an expected SHA256.
Cancellation frees a queue slot only once the submitted work has actually ended.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat
import threading

MAX_READ_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class ReadRequest:
    key: str
    offset: int
    size: int
    sha256: str

    def validate(self):
        if not isinstance(self.key, str) or not self.key or len(self.key) > 1024:
            raise ValueError("invalid request identity")
        if type(self.offset) is not int or type(self.size) is not int or self.offset < 0:
            raise ValueError("invalid file range")
        if not 0 <= self.size <= MAX_READ_BYTES or self.offset + self.size > 2**63 - 1:
            raise ValueError("file range exceeds limits")
        if not isinstance(self.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("expected SHA256 is required")


@dataclass(frozen=True)
class ReadResult:
    request: ReadRequest
    data: bytes


def read_exact(path, request):
    request.validate()
    # Reject known special files before opening. POSIX O_NONBLOCK also prevents
    # a path replaced with a FIFO from hanging between this check and fstat.
    if not stat.S_ISREG(os.stat(path).st_mode):
        raise ValueError("file tier accepts regular files only")
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb", buffering=0) as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("file tier accepts regular files only")
        if request.offset + request.size > info.st_size:
            raise EOFError("requested range exceeds file size")
        stream.seek(request.offset)
        chunks, remaining = [], request.size
        while remaining:
            try:
                data = stream.read(min(remaining, 1024 * 1024))
            except InterruptedError:
                continue
            if not data:
                raise EOFError("short read: file changed or storage failed")
            chunks.append(data)
            remaining -= len(data)
    result = b"".join(chunks)
    if hashlib.sha256(result).hexdigest() != request.sha256:
        raise ValueError("artifact read integrity mismatch")
    return ReadResult(request, result)


class FileTier:
    """At most two pending/active reads; no model state or GPU residency claims."""
    def __init__(self, path, slots=2, reader=read_exact):
        if type(slots) is not int or not 1 <= slots <= 2:
            raise ValueError("slots must be 1 or 2")
        self.path = Path(path)
        self.reader = reader
        self.pool = ThreadPoolExecutor(max_workers=slots, thread_name_prefix="strata-file")
        self.slots = threading.BoundedSemaphore(slots)
        self.lock = threading.Lock()
        self.closed = False

    def submit(self, request):
        request.validate()
        with self.lock:
            if self.closed:
                raise RuntimeError("file tier is closed")
            if not self.slots.acquire(blocking=False):
                raise BufferError("both file-tier slots are busy")
            def perform():
                try:
                    return self.reader(self.path, request)
                finally:
                    # Release BEFORE Future marks completion and wakes result()
                    # callers. A done-callback-only release races a fresh submit.
                    self.slots.release()
            try:
                future = self.pool.submit(perform)
            except BaseException:
                self.slots.release()
                raise
            # A cancelled task never enters perform(); only that path releases
            # here. Running work cannot be cancelled/recycled ahead of completion.
            future.add_done_callback(lambda done: self.slots.release() if done.cancelled() else None)
            return future

    def close(self):
        with self.lock:
            self.closed = True
        self.pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
