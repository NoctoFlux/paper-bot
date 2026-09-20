#!/usr/bin/env python3
"""
report.py - Turn the paper bot's records into a results page you can read on a phone.

Usage:
    python report.py --open              # build paper_state/report.html and open it
    python report.py --watch 60 --open   # rebuild every 60 s while the bot runs (page auto-refreshes)
    python report.py --state-dir other_folder --out my_report.html
    REPORT_PASSWORD=choose-a-password python report.py --serve 8080 --host 0.0.0.0
                                         # small password-protected website (used in the cloud setup)

Reads paper_state/equity.csv and paper_state/trades.csv written by paper_bot.py.
No internet needed; the page is one self-contained HTML file.
"""
import argparse
import base64
import hmac
import html
import json
import math
import os
import sys
import time
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

TARGET = 5.0          # daily target, in percent
LOSS_LIMIT = 2.0      # the bot's daily loss limit, in percent
MARKETS = {
    "US": ("US stocks", "$", "USD"),
    "BIST": ("Borsa Istanbul", "₺", "TRY"),
}

CSS = """
:root{--bg:#EDF1F0;--ink:#13202A;--mute:#5B707D;--rule:#C7D3D6;--gain:#0A8567;--loss:#C93B33;--target:#9A5F08;--wash:#E1E8E8}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0E181E;--ink:#E4EDEF;--mute:#8CA1AC;--rule:#26373F;--gain:#3DCBA0;--loss:#FF7B6F;--target:#E2B24E;--wash:#16252D}}
:root[data-theme="dark"]{--bg:#0E181E;--ink:#E4EDEF;--mute:#8CA1AC;--rule:#26373F;--gain:#3DCBA0;--loss:#FF7B6F;--target:#E2B24E;--wash:#16252D}
html{height:auto;scroll-padding-top:env(safe-area-inset-top,0px)}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  padding:calc(20px + env(safe-area-inset-top,0px)) 18px calc(40px + env(safe-area-inset-bottom,0px));-webkit-text-size-adjust:100%}
main{max-width:620px;margin:0 auto}
h1,h2,.fig{font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;font-weight:600;letter-spacing:-.01em}
h1{font-size:1.7rem;margin:0 0 4px}
h2{font-size:1.25rem;margin:0}
h3{font-size:1rem;margin:26px 0 2px;font-weight:600}
.sub{color:var(--mute);margin:0 0 6px;font-size:.92rem}
.note{background:var(--wash);border-left:3px solid var(--target);padding:8px 12px;font-size:.9rem;margin:14px 0 0;border-radius:0 6px 6px 0}
section.market{margin-top:34px;padding-top:22px;border-top:1px solid var(--rule)}
.head{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
.big{font-size:2.6rem;line-height:1.05;margin:6px 0 0}
.small{color:var(--mute);font-size:.9rem;margin:2px 0 0}
.gain{color:var(--gain)}.loss{color:var(--loss)}.flat{color:var(--mute)}
dl.stats{display:grid;grid-template-columns:1fr 1fr;gap:0 22px;margin:16px 0 0}
dl.stats div{padding:9px 0;border-bottom:1px solid var(--rule)}
dl.stats dt{color:var(--mute);font-size:.82rem}
dl.stats dd{margin:0;font-size:1.05rem;font-variant-numeric:tabular-nums}
svg{display:block;width:100%;height:auto;overflow:visible}
svg text{font-size:13px;fill:var(--mute);font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.ax{stroke:var(--rule);stroke-width:1}
.base{stroke:var(--mute);stroke-width:1;stroke-dasharray:3 4;opacity:.7}
.line{fill:none;stroke-width:2.4;stroke-linejoin:round;stroke-linecap:round}
.line.gain{stroke:var(--gain)}.line.loss{stroke:var(--loss)}.line.flat{stroke:var(--mute)}
.dot.gain{fill:var(--gain)}.dot.loss{fill:var(--loss)}.dot.flat{fill:var(--mute)}
.bar.gain{fill:var(--gain)}.bar.loss{fill:var(--loss)}.bar.flat{fill:var(--mute)}
.halo{paint-order:stroke;stroke:var(--bg);stroke-width:5px;stroke-linejoin:round}
.tline{stroke:var(--target);stroke-width:1.6;stroke-dasharray:6 4}
.ttext{fill:var(--target)!important;font-weight:600}
.lline{stroke:var(--loss);stroke-width:1;stroke-dasharray:1 4;opacity:.8}
.cap{color:var(--mute);font-size:.86rem;margin:8px 0 0}
ul.trades{list-style:none;margin:6px 0 0;padding:0}
ul.trades li{display:grid;grid-template-columns:1fr auto;gap:2px 12px;padding:9px 0;border-bottom:1px solid var(--rule)}
ul.trades .why{color:var(--mute);font-size:.84rem}
ul.trades .pnl{font-variant-numeric:tabular-nums;text-align:right;font-weight:600}
.empty{color:var(--mute);padding:14px 0}
footer{margin-top:38px;color:var(--mute);font-size:.84rem}
"""


def cls(x):
    return "gain" if x > 0.0049 else "loss" if x < -0.0049 else "flat"


def pct(x):
    return f"{x:+.2f}%"


def money(x, sym, decimals=0):
    s = f"{abs(x):,.{decimals}f}"
    return f"{'-' if x < 0 else ''}{sym}{s}"


def signed_money(x, sym):
    return f"{'+' if x > 0 else '-' if x < 0 else ''}{sym}{abs(x):,.2f}"


def read_csv(path):
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def age_text(seconds):
    if seconds < 90:
        return "a moment ago"
    if seconds < 5400:
        return f"{round(seconds / 60)} minutes ago"
    if seconds < 172800:
        return f"{round(seconds / 3600)} hours ago"
    return f"{round(seconds / 86400)} days ago"


def bot_status(state_dir):
    """(text, is_stale) from heartbeat.json, or (None, False) if unavailable."""
    try:
        hb = json.loads((state_dir / "heartbeat.json").read_text())
        last = datetime.fromisoformat(hb["time"])
        age = (datetime.now(timezone.utc) - last).total_seconds()
        return age_text(age), age > max(900, 3 * int(hb.get("poll", 300)))
    except Exception:
        return None, False


# ---------------------------------------------------------------- charts
def equity_svg(equity, start, sym):
    W, H, L, R, T, B = 480, 190, 8, 8, 14, 24
    n = len(equity)
    step = max(1, math.ceil(n / 300))
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    vals = [equity[i] for i in idx]
    lo, hi = min(vals + [start]), max(vals + [start])
    if hi - lo < 1e-9:
        hi = lo + 1
    pad = (hi - lo) * 0.14
    lo, hi = lo - pad, hi + pad

    def X(k):
        return L + (W - L - R) * (k / max(len(vals) - 1, 1))

    def Y(v):
        return T + (H - T - B) * (1 - (v - lo) / (hi - lo))

    pts = " ".join(f"{X(k):.1f},{Y(v):.1f}" for k, v in enumerate(vals))
    c = cls(vals[-1] - start)
    by = Y(start)
    return (
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Account value over time">'
        f'<line class="base" x1="{L}" x2="{W - R}" y1="{by:.1f}" y2="{by:.1f}"/>'
        f'<polyline class="line {c}" points="{pts}"/>'
        f'<text class="halo" x="{L}" y="{by - 6:.1f}">start {money(start, sym)}</text>'
        f'<circle class="dot {c}" cx="{X(len(vals) - 1):.1f}" cy="{Y(vals[-1]):.1f}" r="4"/>'
        f'<line class="ax" x1="{L}" x2="{W - R}" y1="{H - B + 6}" y2="{H - B + 6}"/>'
        f'<text x="{L}" y="{H - 4}">first check</text>'
        f'<text x="{W - R}" y="{H - 4}" text-anchor="end">latest</text>'
        f"</svg>"
    )


def daily_svg(days):
    W, H, L, R, T, B = 480, 190, 36, 8, 12, 22
    vals = days[-40:]
    ymax = max(TARGET + 0.6, max(vals) + 0.3)
    ymin = min(-LOSS_LIMIT - 0.5, min(vals) - 0.3)

    def Y(v):
        return T + (H - T - B) * (1 - (v - ymin) / (ymax - ymin))

    n = len(vals)
    slot = (W - L - R) / n
    bw = min(slot * 0.72, 28)
    bars = []
    for i, v in enumerate(vals):
        x = L + slot * i + (slot - bw) / 2
        y0, y1 = Y(0), Y(v)
        bars.append(f'<rect class="bar {cls(v)}" x="{x:.1f}" y="{min(y0, y1):.1f}" '
                    f'width="{bw:.1f}" height="{max(abs(y1 - y0), 1.5):.1f}" rx="2"/>')
    return (
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Result of each trading day compared with the 5 percent target">'
        f'<line class="tline" x1="{L}" x2="{W - R}" y1="{Y(TARGET):.1f}" y2="{Y(TARGET):.1f}"/>'
        f'<text class="ttext" x="{L - 5}" y="{Y(TARGET) + 4:.1f}" text-anchor="end">5%</text>'
        f'<line class="lline" x1="{L}" x2="{W - R}" y1="{Y(-LOSS_LIMIT):.1f}" y2="{Y(-LOSS_LIMIT):.1f}"/>'
        f'<text x="{L - 5}" y="{Y(-LOSS_LIMIT) + 4:.1f}" text-anchor="end">-{LOSS_LIMIT:g}%</text>'
        + "".join(bars) +
        f'<line class="ax" x1="{L}" x2="{W - R}" y1="{Y(0):.1f}" y2="{Y(0):.1f}"/>'
        f'<text x="{L - 5}" y="{Y(0) + 4:.1f}" text-anchor="end">0</text>'
        f'<text x="{L}" y="{H - 4}">older</text>'
        f'<text x="{W - R}" y="{H - 4}" text-anchor="end">newest day</text>'
        f"</svg>"
    )


# ---------------------------------------------------------------- sections
def market_section(key, eq, tr, mode_demo):
    title, sym, code = MARKETS[key]
    e = eq[eq.market == key].reset_index(drop=True) if not eq.empty else pd.DataFrame()
    if e.empty:
        return (f'<section class="market"><h2>{title}</h2>'
                f'<p class="empty">No results yet. Start paper_bot.py, then run report.py again.</p></section>')

    start = float(e.equity.iloc[0] / (1 + e.total_pct.iloc[0] / 100))
    latest = e.iloc[-1]
    equity = float(latest.equity)
    total = float(latest.total_pct)
    today = float(latest.day_pct)

    peak = e.equity.cummax()
    max_dd = float(((e.equity / peak - 1) * 100).min())

    days = e.groupby("day", sort=False).day_pct.last().tolist()
    hit = sum(1 for d in days if d >= TARGET)

    t = tr[(tr.market == key)] if not tr.empty else pd.DataFrame()
    sells = t[t.side == "SELL"] if not t.empty else pd.DataFrame()
    n_closed = len(sells)
    wins = sells[sells.pnl > 0].pnl.sum() if n_closed else 0.0
    losses = -sells[sells.pnl < 0].pnl.sum() if n_closed else 0.0
    win_rate = f"{(sells.pnl > 0).mean() * 100:.0f}%" if n_closed else "none yet"
    pf_txt = f"{wins / losses:.2f}" if losses > 0 else ("n/a" if n_closed == 0 else "no losses")
    fees = float(t.fee.sum()) if not t.empty else 0.0
    net = float(sells.pnl.sum()) if n_closed else 0.0

    best, worst = max(days), min(days)
    verdict = (f"{hit} of {len(days)} days reached the 5% target. "
               f"Best day {pct(best)}, worst day {pct(worst)}.")

    # recent closed trades
    rows = []
    if n_closed:
        for _, r in sells.iloc[::-1].head(8).iterrows():
            when = pd.to_datetime(r["time"]).strftime("%b %d, %H:%M")
            rows.append(
                f'<li><span>{html.escape(str(r.symbol))} <span class="small">sold {int(r.qty)} at {r.price:,.2f}</span></span>'
                f'<span class="pnl {cls(r.pnl)}">{signed_money(r.pnl, sym)}</span>'
                f'<span class="why">{html.escape(str(r.reason))}</span><span class="why" style="text-align:right">{when}</span></li>')
    trades_html = (f'<h3>Latest closed trades</h3><ul class="trades">{"".join(rows)}</ul>'
                   if rows else '<p class="empty">No closed trades yet.</p>')

    return f"""
<section class="market">
  <div class="head"><h2>{title}</h2><span class="small">{code}</span></div>
  <p class="fig big {cls(total)}">{pct(total)}</p>
  <p class="small">Account value {money(equity, sym)} (started with {money(start, sym)})</p>
  <dl class="stats">
    <div><dt>Today</dt><dd class="{cls(today)}">{pct(today)}</dd></div>
    <div><dt>Wins</dt><dd>{win_rate}</dd></div>
    <div><dt>Closed trades</dt><dd>{n_closed}</dd></div>
    <div><dt>Net trading result</dt><dd class="{cls(net)}">{signed_money(net, sym)}</dd></div>
    <div><dt>Deepest dip</dt><dd>{pct(max_dd)}</dd></div>
    <div><dt>Fees paid</dt><dd>{money(fees, sym)}</dd></div>
    <div><dt>Profit factor</dt><dd>{pf_txt}</dd></div>
    <div><dt>Open positions</dt><dd>{int(latest.open_positions)}</dd></div>
  </dl>
  <h3>Account value</h3>
  {equity_svg(e.equity.tolist(), start, sym)}
  <h3>Each day against the 5% target</h3>
  {daily_svg(days)}
  <p class="cap">{verdict} Compounded, 5% every day would be about 4.3 times your money in 30 trading days.</p>
  {trades_html}
</section>"""


def build(state_dir, refresh):
    eq = read_csv(state_dir / "equity.csv")
    tr = read_csv(state_dir / "trades.csv")
    mode = (state_dir / "mode.txt").read_text().strip() if (state_dir / "mode.txt").exists() else "live"
    demo = mode == "demo"
    stamp = datetime.now().astimezone().strftime("%b %d, %H:%M %Z").strip()
    hb_text, stale = (None, False) if demo else bot_status(state_dir)
    hb_line = f" Bot last checked in {hb_text}." if hb_text else ""
    stale_note = (f'<p class="note">The bot last checked in {hb_text}. It may have stopped, so check the server.</p>'
                  if stale else "")
    refresh_tag = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    demo_note = ('<p class="note">These numbers come from simulated prices (demo mode), so they only show '
                 "how this page looks.</p>") if demo else ""
    body = "".join(market_section(k, eq, tr, demo) for k in MARKETS)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Paper trading results</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E%F0%9F%93%88%3C/text%3E%3C/svg%3E">
{refresh_tag}
<style>{CSS}</style>
</head>
<body>
<main>
  <h1>Paper trading results</h1>
  <p class="sub">Updated {stamp}. Fake money only.{hb_line}</p>
  {demo_note}{stale_note}
  {body}
  <footer>Simulated fills include fees and slippage, but real markets can be worse. Past paper results do not predict real results.</footer>
</main>
</body>
</html>"""


def serve(state_dir, host, port):
    """Tiny password-protected website: the results page plus CSV downloads."""
    user = os.environ.get("REPORT_USER", "me")
    password = os.environ.get("REPORT_PASSWORD", "")
    if len(password) < 8:
        sys.exit("Set REPORT_PASSWORD (8+ characters) before using --serve.")
    token = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, ctype, body, extra=None):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/healthz":
                return self._send(200, "text/plain", "ok")
            got = self.headers.get("Authorization", "")
            if not hmac.compare_digest(got.encode(), token.encode()):
                time.sleep(1)  # slows down password guessing
                return self._send(401, "text/plain", "Password needed",
                                  {"WWW-Authenticate": 'Basic realm="Paper trading"'})
            if path in ("/", "/index.html"):
                return self._send(200, "text/html; charset=utf-8", build(state_dir, 60))
            if path in ("/trades.csv", "/equity.csv"):
                f = state_dir / path.lstrip("/")
                if f.exists():
                    return self._send(200, "text/csv", f.read_bytes(),
                                      {"Content-Disposition": f'attachment; filename="{f.name}"'})
            return self._send(404, "text/plain", "Not found")

        def log_message(self, *args):  # keep logs quiet
            pass

    print(f"Serving results on http://{host}:{port} (user: {user}). Ctrl+C to stop.")
    try:
        ThreadingHTTPServer((host, port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser(description="Build the paper-trading results page")
    ap.add_argument("--state-dir", default="paper_state")
    ap.add_argument("--out", default=None, help="output file (default: <state-dir>/report.html)")
    ap.add_argument("--open", action="store_true", help="open the page in your browser")
    ap.add_argument("--watch", type=int, default=0, help="rebuild every N seconds")
    ap.add_argument("--serve", type=int, default=0, metavar="PORT", help="run as a password-protected website")
    ap.add_argument("--host", default="127.0.0.1", help="address for --serve (0.0.0.0 = reachable from outside)")
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    if args.serve:
        return serve(state_dir, args.host, args.serve)
    out = Path(args.out) if args.out else state_dir / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    opened = False
    while True:
        out.write_text(build(state_dir, args.watch), encoding="utf-8")
        print(f"Wrote {out}")
        if args.open and not opened:
            webbrowser.open(out.resolve().as_uri())
            opened = True
        if not args.watch:
            break
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
