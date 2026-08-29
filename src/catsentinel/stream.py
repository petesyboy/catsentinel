"""Lightweight MJPEG-over-HTTP live view server.

Runs a small HTTP server in a background thread; the capture loop pushes each
frame to it via update_frame(), and any browser that opens http://pi:port/
gets a live <img>-based view (multipart/x-mixed-replace -- understood natively
by every browser, no video codec or plugin needed). Kept to the stdlib
(http.server) + cv2 (already a dependency) rather than pulling in Flask,
since this is the only thing that would need it.

Meant to sit behind a Cloudflare Tunnel (or similar) for remote viewing, not
exposed directly to the internet -- it has no authentication of its own.
"""
from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

_BOUNDARY = "catsentinelframe"

_INDEX_HTML = b"""<!doctype html>
<html><head><title>Cat Sentinel -- live view</title>
<style>body{margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh}
img{max-width:100%;max-height:100%}</style></head>
<body><img src="/stream.mjpg"></body></html>
"""


class _FrameStore:
    """Holds the latest encoded JPEG frame; lets waiters block until a new one arrives."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._version = 0

    def update(self, frame: np.ndarray, quality: int) -> None:
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return
        with self._condition:
            self._jpeg = encoded.tobytes()
            self._version += 1
            self._condition.notify_all()

    def next_frame(self, last_version: int, timeout: float = 5.0) -> tuple[bytes | None, int]:
        """Blocks until a frame newer than last_version is available, or timeout elapses."""
        with self._condition:
            self._condition.wait_for(lambda: self._version != last_version, timeout=timeout)
            return self._jpeg, self._version


def _make_handler(store: _FrameStore):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass  # silence the default per-request access log

        def do_GET(self) -> None:
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(_INDEX_HTML)))
                self.end_headers()
                self.wfile.write(_INDEX_HTML)
                return

            if self.path not in ("/stream", "/stream.mjpg"):
                self.send_response(404)
                self.end_headers()
                return

            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
            self.end_headers()

            last_version = 0
            try:
                while True:
                    jpeg, last_version = store.next_frame(last_version)
                    if jpeg is None:
                        continue
                    self.wfile.write(f"--{_BOUNDARY}\r\n".encode())
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # viewer navigated away/closed the tab -- not an error

    return Handler


class MJPEGStreamServer:
    """Starts a background HTTP server; call update_frame() from the capture loop."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8090, quality: int = 80):
        self._store = _FrameStore()
        self._quality = quality
        self._server = ThreadingHTTPServer((host, port), _make_handler(self._store))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def update_frame(self, frame: np.ndarray) -> None:
        self._store.update(frame, self._quality)

    def stop(self) -> None:
        self._server.shutdown()
        self._thread.join(timeout=5.0)
