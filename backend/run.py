"""Run the API + the built web app on ONE port: python run.py  (from the backend/ directory).

    http://127.0.0.1:8000        web app (served from ../frontend/dist; run `npm run build` after UI changes)
    http://127.0.0.1:8000/docs   API docs
    http://127.0.0.1:8000/api/…  API

`npm run dev` (port 5173) is only needed while editing the frontend; it forwards /api to port 8000.
Another port: PORT=8010 venv/bin/python run.py
"""
import os
import socket
import sys

import uvicorn

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))


def port_free(host: str, port: int) -> bool:
    """True if nothing is listening on host:port (another app there would answer the web page's /api calls)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host if host not in ("0.0.0.0", "") else "127.0.0.1", port)) != 0


if __name__ == "__main__":
    if not port_free(HOST, PORT):
        print(f"\n  Port {PORT} is already used by another program, so Vision AI can't start there.\n"
              f"  Find it:   lsof -iTCP:{PORT} -sTCP:LISTEN\n"
              f"  Stop it (kill <PID> or Ctrl+C in its terminal), or start Vision AI on another port:\n"
              f"             PORT=8010 venv/bin/python run.py\n", file=sys.stderr)
        sys.exit(1)
    print(f"\n  Vision AI → open http://{HOST}:{PORT}   (API docs: http://{HOST}:{PORT}/docs)\n")
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
