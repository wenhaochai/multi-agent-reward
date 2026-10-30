"""A fake SGLang server for tests/dryrun_ext.sh: /health_generate and /health answer 503 for FAKE_LOAD_S seconds
(loading), then 200; /server_info reports world_size 8; /generate answers 200."""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

t0, load = time.time(), float(os.environ.get("FAKE_LOAD_S", "3"))


class H(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        ready = time.time() - t0 >= load
        if self.path in ("/health", "/health_generate"):
            self._send(200 if ready else 503, {})
        elif self.path == "/server_info":
            self._send(200, {"internal_states": [{"world_size": 8}]})
        else:
            self._send(404, {})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._send(200 if self.path == "/generate" else 404, {"meta_info": {}})

    def log_message(self, *a):
        pass


ThreadingHTTPServer((sys.argv[1], int(sys.argv[2])), H).serve_forever()
