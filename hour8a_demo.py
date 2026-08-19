"""Serve the isolated local Hour 8a browser demonstration."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from functools import partial

HOST, PORT = "127.0.0.1", 8765
root = Path(__file__).resolve().parent / "demo"
print(f"Adaptive Tetris demo: http://{HOST}:{PORT}/")
print("Press Ctrl-C to stop.")
ThreadingHTTPServer((HOST, PORT), partial(SimpleHTTPRequestHandler, directory=root)).serve_forever()
