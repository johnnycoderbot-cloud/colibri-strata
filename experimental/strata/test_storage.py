import hashlib
import os
from pathlib import Path
import tempfile
import threading
import unittest
import subprocess
import sys
from concurrent.futures import Future
from unittest.mock import patch

import heads

from storage import FileTier, ReadRequest, read_exact


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "weights.bin"
        self.data = bytes(range(256)) * 100
        self.path.write_bytes(self.data)

    def tearDown(self):
        self.temp.cleanup()

    def request(self, offset=3, size=100, key="layer1-expert2-step3"):
        return ReadRequest(key, offset, size, hashlib.sha256(self.data[offset:offset + size]).hexdigest())

    def test_exact_identity_and_integrity(self):
        request = self.request()
        result = read_exact(self.path, request)
        self.assertEqual(result.request, request)
        self.assertEqual(result.data, self.data[3:103])
        self.assertEqual(read_exact(self.path, self.request(size=0)).data, b"")

    def test_failure_paths(self):
        with self.assertRaises(EOFError):
            read_exact(self.path, self.request(offset=len(self.data), size=1))
        with self.assertRaises(ValueError):
            read_exact(self.path, ReadRequest("x", 0, 100, "0" * 64))
        for request in (ReadRequest("", 0, 1, "a" * 64), ReadRequest("x", -1, 1, "a" * 64),
                        ReadRequest("x", 0, True, "a" * 64), ReadRequest("x", 0, 65 * 1024**2, "a" * 64),
                        ReadRequest("x", 2**63, 0, "a" * 64)):
            with self.assertRaises(ValueError):
                request.validate()

    def test_bound_and_cancel_does_not_reuse_active_slot(self):
        release, entered = threading.Event(), threading.Event()
        def paused(path, request):
            entered.set()
            if not release.wait(10):
                raise TimeoutError("test worker was not released")
            return read_exact(path, request)
        tier = FileTier(self.path, slots=1, reader=paused)
        try:
            future = tier.submit(self.request())
            self.assertTrue(entered.wait(5))
            self.assertFalse(future.cancel())
            with self.assertRaises(BufferError):
                tier.submit(self.request(key="other"))
            release.set()
            self.assertEqual(future.result(timeout=5).data, self.data[3:103])
        finally:
            release.set()
            tier.close()
        with self.assertRaises(RuntimeError):
            tier.submit(self.request())

    def test_reader_failure_releases_slot(self):
        with FileTier(self.path, slots=1) as tier:
            future = tier.submit(ReadRequest("bad", 0, 100, "0" * 64))
            with self.assertRaises(ValueError):
                future.result(timeout=5)
            # Completion callbacks release capacity; a fresh valid read succeeds.
            self.assertEqual(tier.submit(self.request()).result(timeout=5).data, self.data[3:103])

    def test_capacity_is_released_before_future_notification_callbacks(self):
        callbacks_entered, release_callbacks = threading.Event(), threading.Event()
        original = Future._invoke_callbacks
        def paused(future):
            callbacks_entered.set()
            if not release_callbacks.wait(5):
                raise TimeoutError("test callback was not released")
            original(future)
        tier = FileTier(self.path, slots=1)
        with patch.object(Future, "_invoke_callbacks", paused):
            try:
                first = tier.submit(self.request())
                self.assertTrue(callbacks_entered.wait(5))
                self.assertTrue(first.done())
                first.result(timeout=5)
                # A done-callback-only semaphore release fails this submission.
                second = tier.submit(self.request(key="next"))
                release_callbacks.set()
                self.assertEqual(second.result(timeout=5).data, self.data[3:103])
            finally:
                release_callbacks.set()
                tier.close()

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX FIFO test")
    def test_fifo_refused_without_blocking(self):
        fifo = Path(self.temp.name) / "input.fifo"
        os.mkfifo(fifo)
        code = "import sys,hashlib; from storage import read_exact,ReadRequest\ntry:\n read_exact(sys.argv[1],ReadRequest('fifo',0,0,hashlib.sha256(b'').hexdigest()))\nexcept ValueError:\n print('refused'); sys.exit(0)\nsys.exit(1)"
        result = subprocess.run([sys.executable, "-c", code, str(fifo)],
                                cwd=heads.ROOT / "experimental/strata", capture_output=True,
                                text=True, timeout=3)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("refused", result.stdout)


if __name__ == "__main__":
    unittest.main()
