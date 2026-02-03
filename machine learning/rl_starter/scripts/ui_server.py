from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "ui"


class ProcessManager:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: Popen[str] | None = None
        self.command: list[str] | None = None
        self.lines: list[str] = []

    def start(self, command: list[str]) -> bool:
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return False
            self.command = command
            self.lines = []
            self.proc = Popen(command, stdout=PIPE, stderr=PIPE, text=True, cwd=str(ROOT))
            threading.Thread(target=self._drain, args=(self.proc.stdout,), daemon=True).start()
            threading.Thread(target=self._drain, args=(self.proc.stderr,), daemon=True).start()
            return True

    def _drain(self, stream):
        if stream is None:
            return
        for line in stream:
            with self.lock:
                self.lines.append(line.rstrip())
                if len(self.lines) > 500:
                    self.lines = self.lines[-500:]

    def stop(self) -> None:
        with self.lock:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()

    def status(self) -> dict[str, Any]:
        with self.lock:
            if not self.proc:
                return {"state": "idle", "command": None}
            code = self.proc.poll()
            if code is None:
                return {"state": "running", "command": " ".join(self.command or [])}
            return {"state": f"exit({code})", "command": " ".join(self.command or [])}

    def get_logs(self) -> list[str]:
        with self.lock:
            return list(self.lines)


MANAGER = ProcessManager()


def build_command(payload: dict[str, Any]) -> list[str]:
    python = str((ROOT / ".venv" / "Scripts" / "python.exe").resolve())
    if not Path(python).exists():
        python = "python"

    task = payload.get("task")
    game = payload.get("game")
    device = payload.get("device", "auto")

    if task == "train" and game == "lunarlander":
        cmd = [
            python,
            str(ROOT / "scripts" / "train_lunarlander_ppo.py"),
            "--timesteps",
            str(int(payload.get("timesteps", 500000))),
            "--n-envs",
            str(int(payload.get("n_envs", 16))),
            "--device",
            device,
            "--vec-env",
            payload.get("vec_env", "subproc"),
        ]
        if payload.get("vec_normalize"):
            cmd.append("--vec-normalize")
        return cmd

    if task == "eval" and game == "lunarlander":
        return [
            python,
            str(ROOT / "scripts" / "eval_lunarlander_ppo.py"),
            "--episodes",
            str(int(payload.get("episodes", 3))),
            "--device",
            device,
        ]

    if task == "train" and game == "pong":
        return [
            python,
            str(ROOT / "scripts" / "train_pong_dqn.py"),
            "--timesteps",
            str(int(payload.get("timesteps", 1000000))),
            "--n-envs",
            str(int(payload.get("n_envs", 1))),
            "--device",
            device,
        ]

    if task == "eval" and game == "pong":
        return [
            python,
            str(ROOT / "scripts" / "eval_pong_dqn.py"),
            "--episodes",
            str(int(payload.get("episodes", 3))),
            "--device",
            device,
        ]

    raise ValueError("Invalid task/game")


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, data: Any, content_type: str = "application/json") -> None:
        body = data if isinstance(data, (bytes, bytearray)) else json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            content = (UI_DIR / "index.html").read_bytes()
            return self._send(HTTPStatus.OK, content, "text/html")
        if self.path == "/app.js":
            return self._send(HTTPStatus.OK, (UI_DIR / "app.js").read_bytes(), "text/javascript")
        if self.path == "/style.css":
            return self._send(HTTPStatus.OK, (UI_DIR / "style.css").read_bytes(), "text/css")
        if self.path == "/status":
            return self._send(HTTPStatus.OK, MANAGER.status())
        if self.path == "/logs":
            return self._send(HTTPStatus.OK, {"lines": MANAGER.get_logs()})
        return self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = {}
        if length > 0:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))

        if self.path == "/run":
            try:
                cmd = build_command(payload)
                started = MANAGER.start(cmd)
                if not started:
                    return self._send(HTTPStatus.CONFLICT, {"error": "process already running"})
                return self._send(HTTPStatus.OK, {"ok": True, "command": cmd})
            except Exception as exc:
                return self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        if self.path == "/stop":
            MANAGER.stop()
            return self._send(HTTPStatus.OK, {"ok": True})

        return self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})


def main() -> None:
    server = HTTPServer(("127.0.0.1", 8000), Handler)
    print("RL UI running at http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
