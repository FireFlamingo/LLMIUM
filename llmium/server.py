"""Loopback report viewer with a confined read-only report API."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def serve(root, port):
    root = Path(root).resolve()
    page = (Path(__file__).parent / "web" / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            try:
                content_type = "application/json"
                if url.path == "/":
                    body, content_type = page, "text/html; charset=utf-8"
                elif url.path == "/api/reports":
                    body = json.dumps([str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*.json")
                                       if p.name != "llmium-repair.json"]).encode()
                elif url.path == "/api/report":
                    path = (root / parse_qs(url.query)["path"][0]).resolve()
                    if not path.is_relative_to(root) or path.suffix != ".json":
                        raise ValueError("Invalid report path")
                    body = path.read_bytes()
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (OSError, ValueError, KeyError):
                self.send_error(400, "Report unavailable")

    print(f"LLMIUM viewer http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
