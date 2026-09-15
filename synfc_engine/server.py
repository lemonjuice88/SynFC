"""
server.py — thin FastAPI wrapper around the SynFC LangGraph engine.
=========================================================================
Streams each graph step to the browser over Server-Sent Events (SSE),
so the website's Analyst and Agent Interface pages can show the exact
same step-by-step reveal that engine.py's CLI run() prints to a
terminal -- just rendered as a proper UI instead of raw text.

Every request here runs the real LangGraph engine (real DeepSeek API
calls, one run can be dozens of LLM calls) -- so /api/consult is gated
behind an access key (SYNFC_ACCESS_KEY in .env, never committed) and a
light per-key rate limit. Share the key only with people you actually
want spending your API budget.

Run (from this folder, synfc_engine/, same as engine.py's own CLI):
    ../venv/Scripts/python.exe server.py
    (or: ../venv/Scripts/python.exe -m uvicorn server:app --reload --port 8787)

Then point the website at http://localhost:8787 (see js/analyst.js and
js/flow.js) and give visitors the access key out-of-band (not in the
repo -- .env is gitignored).
"""
import json
import os
import queue
import threading
import time

from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

load_dotenv(find_dotenv(usecwd=True))

import engine  # the compiled LangGraph app + ROLE_LABELS live here

ACCESS_KEY = os.environ.get("SYNFC_ACCESS_KEY", "")
if not ACCESS_KEY:
    print("[WARNING] SYNFC_ACCESS_KEY is not set in .env -- /api/consult will "
          "refuse every request until you add one (see .env.example).")

app = FastAPI(title="SynFC Engine API")

# Wide open on purpose: the site may be opened via file:// (origin "null")
# or a plain static server on some other port -- both need to reach this.
# Access control happens at the key check below, not at CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Minimal abuse guard: N requests per key per window. In-memory, so it
# resets on restart -- fine for a small personal deployment; swap for
# something durable (Redis, etc.) if this ever needs to survive restarts
# or run behind multiple worker processes.
# ---------------------------------------------------------------------
RATE_LIMIT = 8
RATE_WINDOW_SECONDS = 10 * 60
_hits: dict = {}


def _rate_limited(key: str) -> bool:
    now = time.time()
    window = _hits.setdefault(key, [])
    window[:] = [t for t in window if now - t < RATE_WINDOW_SECONDS]
    if len(window) >= RATE_LIMIT:
        return True
    window.append(now)
    return False


def _check_key(request: Request):
    """Returns (status_code, message) if the key is missing/wrong/rate
    limited, otherwise None. Kept status-code-agnostic here so each
    endpoint can surface the failure in whatever shape its transport
    needs (plain JSON for a normal fetch(), an SSE 'error' event for
    EventSource -- which can't read a non-200 response body at all)."""
    if not ACCESS_KEY:
        return 503, "Server has no SYNFC_ACCESS_KEY configured."
    supplied = request.query_params.get("key") or request.headers.get("x-synfc-key") or ""
    if supplied != ACCESS_KEY:
        return 401, "Invalid or missing access key."
    if _rate_limited(supplied):
        return 429, "Rate limit reached -- try again later."
    return None


def _sse(event_type: str, payload: dict) -> str:
    data = json.dumps({"type": event_type, **payload}, ensure_ascii=False)
    return f"data: {data}\n\n"


def _run_engine_into_queue(user_message: str, q: "queue.Queue"):
    try:
        initial_state = {"user_message": user_message}
        for step in engine.app.stream(initial_state, stream_mode="updates"):
            for node_name, update in step.items():
                q.put((node_name, update))
    except Exception as e:
        q.put(("__error__", {"message": str(e)}))
    finally:
        q.put(None)  # sentinel: this run is over


def event_stream(user_message: str):
    q: "queue.Queue" = queue.Queue()
    t = threading.Thread(target=_run_engine_into_queue, args=(user_message, q), daemon=True)
    t.start()

    yield _sse("start", {"message": user_message})

    while True:
        item = q.get()  # blocks -- fine, Starlette runs sync generators in a thread pool
        if item is None:
            break
        node_name, update = item

        if node_name == "__error__":
            yield _sse("error", update)
            break
        elif node_name == "mode_classifier":
            yield _sse("mode", {"mode": update.get("mode", "karar")})
        elif node_name == "department_selector":
            yield _sse("router", {
                "roles": update.get("active_roles", []),
                "club_name": update.get("club_name"),
            })
        elif node_name in engine.ROLE_LABELS:
            yield _sse("team_opinion", {
                "team": node_name,
                "label": engine.ROLE_LABELS[node_name],
                "opinion": update.get("role_opinions", {}).get(node_name, ""),
            })
        elif node_name == "collect":
            yield _sse("collect", {
                "conflict": bool(update.get("conflict", False)),
                "conflict_reason": update.get("conflict_reason", ""),
                "active_roles": update.get("active_roles", []),
            })
        elif node_name == "confidence_check":
            # "bilgi" (information) mode ends here -- no debate, no Board.
            yield _sse("information", {"text": update.get("final_decision", "")})
        elif node_name == "debate_dispatch":
            yield _sse("debate_round", {"round": update.get("round_count")})
        elif node_name == "good_cop":
            yield _sse("good_cop", {"text": update.get("board_notes", {}).get("good_cop", "")})
        elif node_name == "bad_cop":
            yield _sse("bad_cop", {"text": update.get("board_notes", {}).get("bad_cop", "")})
        elif node_name == "president":
            yield _sse("president", {"text": update.get("final_decision", "")})

    yield _sse("done", {})


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/verify-key")
def verify_key(request: Request):
    err = _check_key(request)
    if err:
        status_code, msg = err
        return JSONResponse(status_code=status_code, content={"error": msg})
    return {"ok": True}


@app.get("/api/consult")
def consult(request: Request, message: str = ""):
    err = _check_key(request)
    if err:
        _, msg = err

        def denied():
            yield _sse("error", {"message": msg})
        # 200 on purpose: EventSource can't read a non-2xx response body,
        # so an auth/rate-limit failure has to arrive as a normal SSE
        # 'error' event instead of an HTTP error status.
        return StreamingResponse(denied(), media_type="text/event-stream")

    message = (message or "").strip()
    if not message:
        def empty():
            yield _sse("error", {"message": "Empty message"})
        return StreamingResponse(empty(), media_type="text/event-stream")

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # disable proxy buffering, if any sits in front
    }
    return StreamingResponse(event_stream(message), media_type="text/event-stream", headers=headers)


if __name__ == "__main__":
    import uvicorn
    # Render (and most PaaS hosts) inject PORT and expect a bind on
    # 0.0.0.0; falls back to the old local-dev default otherwise.
    port = int(os.environ.get("PORT", 8787))
    host = os.environ.get("HOST", "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
    print(f"SynFC engine API starting on http://{host}:{port}")
    print("Point the website's Analyst / Agent Interface pages at this address (js/config.js).")
    uvicorn.run(app, host=host, port=port)
