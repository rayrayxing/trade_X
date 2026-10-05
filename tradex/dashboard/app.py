"""FastAPI app for the read-only dashboard. GET routes only; the ledger is opened read-only.

Start it with ``tradex dashboard`` (binds 127.0.0.1 unless told otherwise). The page refreshes
through a server-sent-events stream that fires when the ledger changes, with polling as the
fallback if the stream drops.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from tradex.dashboard.views import Sources, Views, clean

STATIC = Path(__file__).resolve().parent / "static"
CSP = ("default-src 'self'; script-src 'self' https://cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline'; "
       "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
POLL_S = 2.0
HEARTBEAT_S = 20.0


def create_app(src: Sources | None = None):
    try:
        from fastapi import FastAPI, HTTPException, Query, Request
        from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:                       # pragma: no cover - message only
        raise SystemExit("the dashboard needs FastAPI: pip install -e '.[dashboard]'") from exc

    views = Views(src or Sources())
    app = FastAPI(title="trade_X dashboard", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.views = views

    @app.middleware("http")
    async def headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers["Content-Security-Policy"] = CSP
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    def ok(data: Any) -> JSONResponse:
        return JSONResponse(clean(data))

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html", media_type="text/html")

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/api/meta")
    def meta():
        return ok(views.meta())

    @app.get("/api/today")
    def today():
        return ok(views.today())

    @app.get("/api/positions")
    def positions(at: str | None = None, book: str = "ensemble"):
        return ok(views.positions(at=at, book=book))

    @app.get("/api/decisions")
    def decisions(limit: int = Query(100, ge=1, le=500), outcome: str | None = None, symbol: str | None = None):
        return ok(views.decisions(limit=limit, outcome=outcome, symbol=symbol))

    @app.get("/api/decisions/{decision_id}")
    def decision(decision_id: str):
        d = views.decision(decision_id)
        if d is None:
            raise HTTPException(404, "no such decision")
        return ok(d)

    @app.get("/api/bars/{symbol}")
    def bars(symbol: str, tf: str = "D1", decision_id: str | None = None):
        return ok(views.bars(symbol, tf, decision_id))

    @app.get("/api/performance")
    def performance():
        return ok(views.performance())

    @app.get("/api/strategies")
    def strategies():
        return ok(views.strategies())

    @app.get("/api/system")
    def system():
        return ok(views.system())

    @app.get("/api/readiness")
    def readiness():
        return ok(views.readiness())

    @app.get("/api/stream")
    async def stream(once: bool = False):
        """Server-sent events: one ``change`` event whenever the ledger moves, a comment every 20 s."""
        async def gen():
            last, idle = None, 0.0
            while True:
                tok = await asyncio.to_thread(views.change_token)
                if tok != last:
                    last, idle = tok, 0.0
                    yield f"event: change\ndata: {tok}\n\n"
                    if once:
                        return
                elif idle >= HEARTBEAT_S:
                    idle = 0.0
                    yield ": keepalive\n\n"
                await asyncio.sleep(POLL_S)
                idle += POLL_S
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def serve(src: Sources, host: str = "127.0.0.1", port: int = 8765) -> int:
    try:
        import uvicorn
    except ImportError:                              # pragma: no cover
        raise SystemExit("the dashboard needs uvicorn: pip install -e '.[dashboard]'")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"warning: binding {host}: the dashboard has no login. Only do this on a network you trust "
              "(or put it behind Tailscale).")
    uvicorn.run(create_app(src), host=host, port=port, log_level="info")
    return 0
