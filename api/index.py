import os
import sys
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import btc_score  # noqa: E402

HEAD = ('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">')


def seconds_to_refresh(now):
    # Binance daily candle closes 00:00 UTC; recompute shortly after.
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=15, second=0, microsecond=0)
    if now.hour == 0 and now.minute < 15:
        nxt -= timedelta(days=1)
    return max(60, int((nxt - now).total_seconds()))


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            body = (HEAD + btc_score.page()[3]).encode()
            status, cache = 200, f"public, max-age=0, s-maxage={seconds_to_refresh(datetime.now(timezone.utc))}"
        except Exception as e:  # data source down: don't let the CDN cache the error
            body = f"{HEAD}<p style='font-family:system-ui;padding:16px'>Data source unavailable ({type(e).__name__}). Try again in a few minutes.</p>".encode()
            status, cache = 502, "no-store"
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)
