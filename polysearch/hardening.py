"""Bounded request handling for the JSON-only local and server API."""
from starlette.responses import JSONResponse
from starlette.datastructures import Headers

DEFAULT_BODY_LIMIT = 256 * 1024
CSV_BODY_LIMIT = 8 * 1024 * 1024


class RequestBounds:
    """Pure ASGI middleware: check both declared size and streamed bytes.

    Buffers at most the route budget, then replays the body to FastAPI.
    JSON-only POST requests; empty logout is permitted.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        limit = CSV_BODY_LIMIT if scope["path"].rstrip("/") == "/api/import.csv" else DEFAULT_BODY_LIMIT

        async def reject(status, detail):
            await JSONResponse({"detail": detail}, status_code=status,
                               headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})(scope, receive, send)

        lengths = headers.getlist("content-length")
        if len(lengths) > 1:
            await reject(400, "Ambiguous Content-Length")
            return
        if lengths:
            try:
                declared = int(lengths[0])
                if declared < 0:
                    raise ValueError()
            except ValueError:
                await reject(400, "Invalid Content-Length")
                return
            if declared > limit:
                await reject(413, "Request body too large")
                return
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > limit:
                await reject(413, "Request body too large")
                return
            if chunk:
                chunks.append(chunk)
            if not message.get("more_body", False):
                break
        if lengths and size != declared:
            await reject(400, "Content-Length does not match body")
            return
        if size and headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
            await reject(415, "Expected application/json")
            return
        first = True

        async def replay():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
