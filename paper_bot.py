#!/usr/bin/env python3
"""
paper_bot.py - Paper-trading bot for US stocks and Borsa Istanbul (BIST).

FAKE MONEY ONLY. No broker connection, no real orders.

Setup:
    pip install pandas numpy yfinance

Usage:
    python paper_bot.py --demo                 # offline test with synthetic prices
    python paper_bot.py                        # live data, both markets, poll every 5 min
    python paper_bot.py --markets US           # only US
    python paper_bot.py --interval 15m --poll 300
    python paper_bot.py --reset                # wipe saved state and start fresh

    python paper_bot.py --once                 # one check, then exit (used by GitHub Actions)

Then view your results with:  python report.py --open

Strategy (deliberately simple): fast/slow moving-average crossover, ATR stop-loss.
Risk controls: per-trade risk %, max position size, max open positions,
daily loss limit (flattens and halts), optional flatten before close.

The "5% daily target" is only TRACKED and reported. The bot never takes extra
risk to reach it - that is how accounts blow up.

Limitations: exchange holidays are only partly handled (stale-data guard),
yfinance data is delayed and can have gaps, fills are simulated.
"""
import argparse
import csv
import json
import math
import signal
import time
import traceback
from datetime import datetime, timezone, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
CFG = {
    "fast": 10,                 # fast SMA length (bars)
    "slow": 30,                 # slow SMA length (bars)
    "atr_n": 14,                # ATR length
    "atr_mult": 2.0,            # stop = entry - atr_mult * ATR
    "risk_per_trade": 0.01,     # risk 1% of equity per trade
    "max_position_pct": 0.20,   # never put more than 20% of equity in one stock
    "max_positions": 4,
    "daily_loss_limit": 0.02,   # flatten + halt for the day at -2%
    "daily_target": 0.05,       # tracked only, never chased
    "flatten_before_close": True,
    "flatten_minutes": 10,      # minutes before close to flatten
    "demo_bars_per_day": 26,
}

MARKETS = {
    "US": {
        "tz": "America/New_York", "open": (9, 30), "close": (16, 0),
        "currency": "USD", "start_cash": 10_000.0,
        "commission": 0.0000, "slippage": 0.0005,
        "symbols": ["AAPL", "MSFT", "NVDA", "AMZN", "SPY"],
    },
    "BIST": {
        "tz": "Europe/Istanbul", "open": (10, 0), "close": (18, 0),
        "currency": "TRY", "start_cash": 300_000.0,
        "commission": 0.0015, "slippage": 0.0010,
        "symbols": ["THYAO.IS", "GARAN.IS", "AKBNK.IS", "ASELS.IS", "EREGL.IS"],
    },
}


# --------------------------------------------------------------------------
# Data feeds
# --------------------------------------------------------------------------
class LiveFeed:
    """Delayed market data from Yahoo Finance via yfinance."""

    def __init__(self, interval):
        self.interval = interval
        mins = int(interval[:-1]) * (60 if interval.endswith("h") else 1)
        self.stale_after = max(45, 3 * mins)  # minutes

    def bars(self, sym):
        import yfinance as yf
        try:
            df = yf.download(sym, period="5d", interval=self.interval,
                             progress=False, auto_adjust=True)
        except Exception as e:  # network problems etc.
            print(f"  ! data error for {sym}: {e}")
            return None
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["High", "Low", "Close"]].dropna()
        if df.empty:
            return None
        last = df.index[-1]
        if last.tzinfo is None:
            last = last.tz_localize("UTC")
        age_min = (pd.Timestamp.now(tz="UTC") - last).total_seconds() / 60
        if age_min > self.stale_after:  # holiday / halted / feed problem
            return None
        return df


class DemoFeed:
    """Synthetic random-walk prices so you can test everything offline."""

    def __init__(self, symbols, n=5000, seed=7):
        self.step = 0
        rng = np.random.default_rng(seed)
        self.data = {}
        for s in symbols:
            close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.006, n)))
            high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
            low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
            self.data[s] = pd.DataFrame({"High": high, "Low": low, "Close": close})

    def bars(self, sym):
        return self.data[sym].iloc[: 60 + self.step]


# --------------------------------------------------------------------------
# Strategy
# --------------------------------------------------------------------------
def analyze(df):
    c = df["Close"]
    fast = c.rolling(CFG["fast"]).mean()
    slow = c.rolling(CFG["slow"]).mean()
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - c.shift()).abs(),
                    (df["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(CFG["atr_n"]).mean()
    return {
        "price": float(c.iloc[-1]),
        "atr": float(atr.iloc[-1]),
        "bull": bool(fast.iloc[-1] > slow.iloc[-1]),
        "cross_up": bool(fast.iloc[-2] <= slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1]),
    }


# --------------------------------------------------------------------------
# Paper portfolio
# --------------------------------------------------------------------------
class Portfolio:
    def __init__(self, name, cfg, trade_log):
        self.name, self.cfg, self.trade_log = name, cfg, trade_log
        self.cash = cfg["start_cash"]
        self.positions = {}          # sym -> {qty, entry, entry_fee, stop}
        self.day = None
        self.day_start = cfg["start_cash"]
        self.halted = False
        self.total_fees = 0.0

    def equity(self, prices):
        return self.cash + sum(p["qty"] * prices.get(s, p["entry"])
                               for s, p in self.positions.items())

    def buy(self, sym, price, qty, stop):
        fill = price * (1 + self.cfg["slippage"])
        fee = fill * qty * self.cfg["commission"]
        self.cash -= fill * qty + fee
        self.total_fees += fee
        self.positions[sym] = {"qty": qty, "entry": fill, "entry_fee": fee, "stop": stop}
        self._log(sym, "BUY", qty, fill, fee, "signal", 0.0)

    def sell(self, sym, price, reason):
        p = self.positions.pop(sym)
        fill = price * (1 - self.cfg["slippage"])
        fee = fill * p["qty"] * self.cfg["commission"]
        self.cash += fill * p["qty"] - fee
        self.total_fees += fee
        pnl = (fill - p["entry"]) * p["qty"] - fee - p["entry_fee"]
        self._log(sym, "SELL", p["qty"], fill, fee, reason, pnl)

    def log_equity(self, eq):
        """One row per check, used by report.py for the charts."""
        path = self.trade_log.with_name("equity.csv")
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "market", "day", "equity", "day_pct", "total_pct", "open_positions"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), self.name, self.day,
                        round(eq, 2), round((eq / self.day_start - 1) * 100, 3),
                        round((eq / self.cfg["start_cash"] - 1) * 100, 3), len(self.positions)])

    def _log(self, sym, side, qty, price, fee, reason, pnl):
        new = not self.trade_log.exists()
        with self.trade_log.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "market", "symbol", "side", "qty", "price", "fee", "reason", "pnl"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), self.name, sym, side,
                        qty, round(price, 4), round(fee, 4), reason, round(pnl, 2)])
        tag = f" pnl {pnl:+.2f}" if side == "SELL" else ""
        print(f"  {side:4} {qty} {sym} @ {price:.2f} ({reason}){tag}")

    def save(self, path):
        keys = ["cash", "positions", "day", "day_start", "halted", "total_fees"]
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: getattr(self, k) for k in keys}, indent=2))
        tmp.replace(path)  # atomic swap

    def load(self, path):
        if path.exists():
            for k, v in json.loads(path.read_text()).items():
                setattr(self, k, v)


# --------------------------------------------------------------------------
# Market session logic
# --------------------------------------------------------------------------
def session(cfg, step, demo):
    """Returns (day_key, is_open, near_close)."""
    if demo:
        n = CFG["demo_bars_per_day"]
        return f"demo-{step // n}", True, (step % n) >= n - 2
    now = datetime.now(ZoneInfo(cfg["tz"]))
    o, c = dtime(*cfg["open"]), dtime(*cfg["close"])
    is_open = now.weekday() < 5 and o <= now.time() < c
    close_dt = datetime.combine(now.date(), c, tzinfo=now.tzinfo)
    near = (close_dt - now).total_seconds() <= CFG["flatten_minutes"] * 60
    return now.date().isoformat(), is_open, near


# --------------------------------------------------------------------------
# One decision cycle for one market
# --------------------------------------------------------------------------
def run_tick(name, cfg, pf, feed, step, demo, say_closed=False):
    day, is_open, near_close = session(cfg, step, demo)
    if not is_open:
        if say_closed:
            print(f"[{name}] market closed - nothing to do")
        return

    sigs = {}
    for sym in cfg["symbols"]:
        df = feed.bars(sym)
        if df is not None and len(df) >= CFG["slow"] + 2:
            s = analyze(df)
            if not (math.isnan(s["atr"]) or s["atr"] <= 0):
                sigs[sym] = s
    if not sigs:
        print(f"[{name}] no fresh data (holiday or feed issue) - skipping")
        return
    prices = {s: v["price"] for s, v in sigs.items()}

    eq = pf.equity(prices)
    if pf.day != day:                      # new trading day
        pf.day, pf.day_start, pf.halted = day, eq, False

    # 1) Daily loss limit: flatten and stop for the day
    if not pf.halted and eq <= pf.day_start * (1 - CFG["daily_loss_limit"]):
        print(f"[{name}] DAILY LOSS LIMIT hit - flattening and halting")
        for sym in list(pf.positions):
            if sym in prices:
                pf.sell(sym, prices[sym], "daily loss limit")
        pf.halted = True

    # 2) Exits
    for sym in list(pf.positions):
        if sym not in sigs:
            continue
        s, p = sigs[sym], pf.positions[sym]
        if s["price"] <= p["stop"]:
            pf.sell(sym, s["price"], "stop-loss")
        elif not s["bull"]:
            pf.sell(sym, s["price"], "trend exit")
        elif near_close and CFG["flatten_before_close"]:
            pf.sell(sym, s["price"], "end of day")

    # 3) Entries
    if not pf.halted and not near_close:
        for sym, s in sigs.items():
            if sym in pf.positions or len(pf.positions) >= CFG["max_positions"]:
                continue
            if not s["cross_up"]:
                continue
            eq = pf.equity(prices)
            stop_dist = CFG["atr_mult"] * s["atr"]
            unit_cost = s["price"] * (1 + cfg["slippage"] + cfg["commission"])
            qty = math.floor(min(eq * CFG["risk_per_trade"] / stop_dist,
                                 eq * CFG["max_position_pct"] / s["price"],
                                 pf.cash / unit_cost))
            if qty >= 1:
                pf.buy(sym, s["price"], qty, s["price"] - stop_dist)

    # 4) Status
    eq = pf.equity(prices)
    day_pct = (eq / pf.day_start - 1) * 100
    total_pct = (eq / cfg["start_cash"] - 1) * 100
    pf.log_equity(eq)
    flag = " HALTED" if pf.halted else ""
    print(f"[{name}] equity {eq:,.2f} {cfg['currency']} | today {day_pct:+.2f}% "
          f"(target {CFG['daily_target']*100:.0f}%) | total {total_pct:+.2f}% | "
          f"open {len(pf.positions)} | fees {pf.total_fees:,.2f}{flag}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Paper-trading bot (US + BIST)")
    ap.add_argument("--markets", default="US,BIST")
    ap.add_argument("--interval", default="15m", help="bar size: 5m, 15m, 30m, 1h")
    ap.add_argument("--poll", type=int, default=300, help="seconds between checks")
    ap.add_argument("--demo", action="store_true", help="synthetic data, runs instantly")
    ap.add_argument("--steps", type=int, default=400, help="demo bars to simulate")
    ap.add_argument("--once", action="store_true",
                    help="do one check per market, then exit (for schedulers like GitHub Actions)")
    ap.add_argument("--reset", action="store_true", help="delete saved state")
    ap.add_argument("--state-dir", default="paper_state")
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    names = [m.strip().upper() for m in args.markets.split(",")]
    trade_log = state_dir / "trades.csv"

    if args.reset:
        for f in state_dir.glob("*"):
            f.unlink()

    (state_dir / "mode.txt").write_text("demo" if args.demo else "live")

    pfs = {}
    for n in names:
        pfs[n] = Portfolio(n, MARKETS[n], trade_log)
        pfs[n].load(state_dir / f"{n}.json")

    feed = (DemoFeed([s for n in names for s in MARKETS[n]["symbols"]])
            if args.demo else LiveFeed(args.interval))
    print(f"Paper trading {', '.join(names)} ({'DEMO' if args.demo else 'LIVE DATA'}). Ctrl+C to stop.\n")

    if args.once:  # one pass and exit; the scheduler calls us again later
        for n in names:
            try:
                run_tick(n, MARKETS[n], pfs[n], feed, 0, args.demo, say_closed=True)
                pfs[n].save(state_dir / f"{n}.json")
            except Exception:
                print(f"[{n}] error during check:")
                traceback.print_exc()
        return

    # `docker stop` / systemd send SIGTERM: treat it like Ctrl+C so state is saved cleanly
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    step = 0
    try:
        while True:
            for n in names:
                try:
                    run_tick(n, MARKETS[n], pfs[n], feed, step, args.demo)
                    pfs[n].save(state_dir / f"{n}.json")
                except Exception:  # never let one bad check kill a 24/7 bot
                    print(f"[{n}] error during check, will retry next round:")
                    traceback.print_exc()
            # heartbeat: lets report.py show whether the bot is alive
            hb = state_dir / "heartbeat.json"
            hb.with_suffix(".tmp").write_text(json.dumps(
                {"time": datetime.now(timezone.utc).isoformat(), "poll": args.poll}))
            hb.with_suffix(".tmp").replace(hb)
            step += 1
            if args.demo:
                feed.step = step
                if step >= args.steps:
                    break
            else:
                time.sleep(args.poll)
    except KeyboardInterrupt:
        print("\nStopped.")
    for n in names:
        pfs[n].save(state_dir / f"{n}.json")
    print(f"\nTrades logged in {trade_log}")
    print(f"See your results: python report.py --state-dir {state_dir} --open")


if __name__ == "__main__":
    main()
