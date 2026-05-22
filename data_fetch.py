"""
data_fetch.py — All market data fetching, caching, and live-tail helpers.
"""
import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import polars as pl
    _POLARS_OK = True
except ImportError:
    _POLARS_OK = False

try:
    import aiohttp
    _AIOHTTP_OK = True
except ImportError:
    _AIOHTTP_OK = False

try:
    import pyarrow
    _PARQUET_OK = True
except ImportError:
    _PARQUET_OK = False

from config import (
    MODE_CFG, VIX_CALM, VIX_CAUTION, VIX_STRESS,
    NSE_OPEN_HOUR, NSE_OPEN_MIN, NSE_CLOSE_HOUR, NSE_CLOSE_MIN,
    _CACHE_DIR,
)
from indicators import ema, action_label

def _cache_path(sym: str, interval: str) -> Path:
    return _CACHE_DIR / f"{sym.replace('.NS','').upper()}_{interval}.parquet"

# ══════════════════════════════════════════════════════════════════════════════
# Key change: call _normalize_index so the returned DataFrame always has a
# tz-aware DatetimeIndex regardless of how pyarrow or Polars reconstructed it.
# ══════════════════════════════════════════════════════════════════════════════

def _load_cached(sym: str, interval: str) -> Optional[pd.DataFrame]:
    """Load cached historical bars; always returns a tz-aware DatetimeIndex."""
    p = _cache_path(sym, interval)
    if not p.exists():
        return None
    try:
        if _POLARS_OK:
            # Polars may strip tz info on round-trip; _normalize_index fixes it
            df = pl.read_parquet(p).to_pandas()
        else:
            df = pd.read_parquet(p)
        return _normalize_index(df)
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# NEW HELPER — insert once, just above _save_cached
# ══════════════════════════════════════════════════════════════════════════════
_IST = "Asia/Kolkata"

def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee that df has a tz-aware DatetimeIndex in Asia/Kolkata.

    Handles three situations that arise at the cache/fetch boundary:
      1. Index is already a tz-aware DatetimeIndex (live fetch path) → convert tz.
      2. Index is a tz-naive DatetimeIndex (Polars round-trip strips tz) → localize.
      3. Index is RangeIndex and a datetime column exists (save/load mismatch) → promote.
    """
    if df is None or df.empty:
        return df

    # ── Case 3: index column was saved as a data column ──────────────────────
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ("ts", "Datetime", "datetime", "index", "Date", "date"):
            if col in df.columns:
                df = df.copy()
                series = pd.to_datetime(df[col], errors="coerce")
                if series.dt.tz is None:
                    series = series.dt.tz_localize("UTC").dt.tz_convert(_IST)
                else:
                    series = series.dt.tz_convert(_IST)
                df.index = series
                df = df.drop(columns=[col])
                break

    # ── Cases 1 & 2: fix timezone on existing DatetimeIndex ──────────────────
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(_IST)
        elif str(df.index.tz) != _IST:
            df.index = df.index.tz_convert(_IST)

    return df
# ══════════════════════════════════════════════════════════════════════════════
# Key change: save the timestamp column as "ts" (predictable name) instead of
# relying on the default "index" name that reset_index() produces.
# ══════════════════════════════════════════════════════════════════════════════
def _save_cached(sym: str, interval: str, df: pd.DataFrame):
    """Save historical bars to parquet with a reliable 'ts' timestamp column."""
    if not _PARQUET_OK:
        return
    try:
        p       = _cache_path(sym, interval)
        df_save = df.copy()
        if isinstance(df_save.index, pd.DatetimeIndex):
            df_save.index.name = "ts"          # always 'ts', never unnamed "index"
            df_save = df_save.reset_index()
        df_save.to_parquet(p, index=False, compression="snappy")
    except Exception:
        pass


def _cache_is_fresh(sym: str, interval: str, max_age_hours: float) -> bool:
    p = _cache_path(sym, interval)
    if not p.exists():
        return False
    age = (time.time() - p.stat().st_mtime) / 3600
    return age < max_age_hours

def _is_market_open() -> bool:
    now_ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
    if now_ist.weekday() >= 5:
        return False
    minutes = now_ist.hour * 60 + now_ist.minute
    open_m  = NSE_OPEN_HOUR * 60 + NSE_OPEN_MIN
    close_m = NSE_CLOSE_HOUR * 60 + NSE_CLOSE_MIN
    return open_m <= minutes <= close_m

def _cold_start_needed(mode: str) -> bool:
    """True if we haven't done a full historical fetch today."""
    flag = _CACHE_DIR / f"cold_start_{mode}_{datetime.utcnow().date()}.flag"
    return not flag.exists()

def _mark_cold_start_done(mode: str):
    flag = _CACHE_DIR / f"cold_start_{mode}_{datetime.utcnow().date()}.flag"
    flag.touch()

# ══════════════════════════════════════════════════════════════════════════════
# SPEED-1: ASYNC HTTP  — Direct Yahoo Finance v8 API
# ══════════════════════════════════════════════════════════════════════════════

_YF_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
_YF_HDR  = {
    "User-Agent": "Mozilla/5.0 (compatible; BullSutra/15)",
    "Accept":     "application/json",
}

def _yf_period_to_range(period: str):
    """Map yfinance period string → (range1, range2) for Yahoo v8 API."""
    _map = {
        "5d":"5d","1mo":"1mo","3mo":"3mo","6mo":"6mo",
        "1y":"1y","2y":"2y","5y":"5y","ytd":"ytd","max":"max",
    }
    return _map.get(period, "1y")

def _parse_yahoo_v8(data: dict, sym: str) -> pd.DataFrame:
    """Parse Yahoo v8 JSON response → OHLCV DataFrame."""
    try:
        res    = data["chart"]["result"][0]
        ts     = res["timestamp"]
        ohlcv  = res["indicators"]["quote"][0]
        adj    = res["indicators"].get("adjclose", [{}])[0].get("adjclose", ohlcv["close"])
        idx    = pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Kolkata")
        df     = pd.DataFrame({
            "Open":   ohlcv["open"],
            "High":   ohlcv["high"],
            "Low":    ohlcv["low"],
            "Close":  adj if adj else ohlcv["close"],
            "Volume": ohlcv["volume"],
        }, index=idx)
        df["Volume"] = df["Volume"].fillna(0)
        return df.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()

async def _fetch_one_async(session: "aiohttp.ClientSession",
                           sym: str, period: str, interval: str) -> tuple[str, pd.DataFrame]:
    ticker = sym if sym.endswith(".NS") else sym + ".NS"
    url    = f"{_YF_BASE}/{ticker}"
    params = {"range": _yf_period_to_range(period), "interval": interval,
               "includeAdjustedClose": "true", "events": ""}
    for attempt in range(3):
        try:
            async with session.get(url, params=params, headers=_YF_HDR,
                                   timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    return sym, _parse_yahoo_v8(data, sym)
        except Exception:
            pass
        await asyncio.sleep(0.3 * (attempt + 1))
    return sym, pd.DataFrame()

async def _fetch_live_tail_async(session: "aiohttp.ClientSession",
                                 sym: str, interval: str,
                                 n_bars: int = 3) -> tuple[str, pd.DataFrame]:
    """Fetch only the last n_bars (for live refresh during session)."""
    ticker = sym if sym.endswith(".NS") else sym + ".NS"
    url    = f"{_YF_BASE}/{ticker}"
    period = "1d" if interval in ("1m","5m","15m","30m") else "5d"
    params = {"range": period, "interval": interval,
               "includeAdjustedClose": "true", "events": ""}
    for attempt in range(2):
        try:
            async with session.get(url, params=params, headers=_YF_HDR,
                                   timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    df   = _parse_yahoo_v8(data, sym)
                    if not df.empty:
                        return sym, df.iloc[-n_bars:]
        except Exception:
            pass
        await asyncio.sleep(0.3)
    return sym, pd.DataFrame()

async def _batch_fetch_async(symbols: list, period: str, interval: str,
                              concurrency: int = 64) -> dict[str, pd.DataFrame]:
    """Fetch all symbols concurrently using a shared aiohttp session."""
    results: dict[str, pd.DataFrame] = {}
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(sym):
        async with sem:
            return await _fetch_one_async(session, sym, period, interval)

    connector = aiohttp.TCPConnector(limit=concurrency, ttl_dns_cache=300,
                                     enable_cleanup_closed=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [asyncio.create_task(_bounded(s)) for s in symbols]
        for coro in asyncio.as_completed(tasks):
            sym, df = await coro
            results[sym] = df
    return results

def fetch_async(symbols: list, period: str, interval: str,
                concurrency: int = 64) -> dict[str, pd.DataFrame]:
    """Sync wrapper: run async fetch in a new event loop (thread-safe)."""
    if not _AIOHTTP_OK:
        return _yf_fallback_batch(symbols, period, interval)
    try:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                _batch_fetch_async(symbols, period, interval, concurrency)
            )
        finally:
            loop.close()
    except Exception:
        return _yf_fallback_batch(symbols, period, interval)

def _yf_fallback_batch(symbols: list, period: str, interval: str) -> dict[str, pd.DataFrame]:
    """Fallback: use yfinance batch download if aiohttp unavailable."""
    import yfinance as yf
    tickers = [s if s.endswith(".NS") else s+".NS" for s in symbols]
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i+50]
        try:
            raw = yf.download(batch, period=period, interval=interval,
                              auto_adjust=True, progress=False, threads=False,
                              group_by="ticker")
            for sym, tkr in zip(symbols[i:i+50], batch):
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        if tkr in raw.columns.get_level_values(0):
                            df = raw[tkr].copy()
                        elif tkr in raw.columns.get_level_values(1):
                            df = raw.xs(tkr, axis=1, level=1).copy()
                        else:
                            df = pd.DataFrame()
                    else:
                        df = raw.copy()
                    df["Volume"] = df["Volume"].fillna(0) if not df.empty else df
                    out[sym] = df.dropna(subset=["Close"]) if not df.empty else pd.DataFrame()
                except Exception:
                    out[sym] = pd.DataFrame()
        except Exception:
            for sym in symbols[i:i+50]:
                out[sym] = pd.DataFrame()
    return out

# ══════════════════════════════════════════════════════════════════════════════
# SPEED-3: INCREMENTAL FETCH  — cache + live-tail merge
# ══════════════════════════════════════════════════════════════════════════════




def batch_incremental_fetch(
    symbols: list,
    mode: str,
    force_full: bool = False,
    progress_cb=None,
) -> dict:
    cfg      = MODE_CFG[mode]
    interval = cfg["interval"]
    period   = cfg["yf_period"]
    min_bars = cfg["hist_min_bars"]

    need_full  = []
    can_append = []
    results    = {}

    for sym in symbols:
        c = _load_cached(sym, interval)
        if force_full or c is None or len(c) < min_bars:
            need_full.append(sym)
        else:
            can_append.append(sym)

    total = len(symbols)

    if need_full:
        fresh = fetch_async(need_full, period, interval, concurrency=20)
        for sym, df in fresh.items():
            if not df.empty:
                _save_cached(sym, interval, df)
                results[sym] = df
            else:
                results[sym] = pd.DataFrame()
        if progress_cb:
            progress_cb(len(need_full) / total)

    # Cache staleness thresholds (seconds) — market-closed path also enforces these
    _STALE_SECS = {"Intraday": 3600, "Swing": 43200, "Positional": 86400}
    _stale_secs = _STALE_SECS.get(mode, 43200)

    if can_append and _is_market_open():
        live_int = cfg.get("live_interval", interval)
        tails    = fetch_async(can_append, "1d", live_int, concurrency=20)
        done     = len(need_full)
        for sym in can_append:
            cached = _load_cached(sym, interval)
            tail   = tails.get(sym, pd.DataFrame())
            if cached is not None and not tail.empty:
                cached = _normalize_index(cached)
                tail   = _normalize_index(tail)
                merged = pd.concat([cached, tail])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                _save_cached(sym, interval, merged)
                results[sym] = merged
            elif cached is not None:
                results[sym] = cached
            else:
                results[sym] = pd.DataFrame()
            done += 1
            if progress_cb:
                progress_cb(done / total)
    else:
        # FIX-2: enforce staleness even when market is closed; re-fetch stale caches
        stale_syms = [
            sym for sym in can_append
            if not _cache_is_fresh(sym, interval, _stale_secs / 3600)
        ]
        fresh_syms = [sym for sym in can_append if sym not in stale_syms]

        if stale_syms:
            refreshed = fetch_async(stale_syms, period, interval, concurrency=20)
            for sym, df in refreshed.items():
                if not df.empty:
                    _save_cached(sym, interval, df)
                results[sym] = df if not df.empty else pd.DataFrame()

        for sym in fresh_syms:
            cached = _load_cached(sym, interval)
            results[sym] = cached if cached is not None else pd.DataFrame()
        if progress_cb:
            progress_cb(1.0)

    return results

# ══════════════════════════════════════════════════════════════════════════════
# SPEED-4 + SPEED-5: VECTORIZED BATCH INDICATORS (numpy)
# ══════════════════════════════════════════════════════════════════════════════


@st.cache_data(ttl=300)
def fetch_vix():
    try:
        raw = fetch_async(["^INDIAVIX"], "5d", "1d", concurrency=1)
        df  = raw.get("^INDIAVIX", pd.DataFrame())
        if df.empty:
            import yfinance as yf
            df = yf.download("^INDIAVIX", period="5d", interval="1d",
                             auto_adjust=True, progress=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
        if df.empty:
            return None, "UNKNOWN"
        v = float(df["Close"].iloc[-1])
        label = "CALM" if v < VIX_CALM else ("CAUTION" if v < VIX_CAUTION else "STRESS")
        return round(v, 2), label
    except Exception:
        return None, "UNKNOWN"


def _htf_cache_path(ticker: str, interval: str) -> Path:
    """Separate subdirectory keeps HTF parquets away from primary-TF cache."""
    d = _CACHE_DIR / "htf"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{ticker.replace('.NS','').upper()}_{interval}_htf.parquet"

_HTF_TTL = {"Intraday": 4, "Swing": 72, "Positional": 240}  # hours per mode

def _fetch_htf_cached(ticker: str, period: str, interval: str,
                      mode: str = "Swing") -> pd.DataFrame:
    """
    Disk-backed HTF cache — survives page refresh and process restarts.
    TTL: 4 h (Intraday) / 72 h (Swing) / 240 h (Positional).
    Falls back to live fetch on miss or expiry.
    """
    p = _htf_cache_path(ticker, interval)
    ttl_h = _HTF_TTL.get(mode, 72)

    # ── Cache hit ──────────────────────────────────────────────────────────
    if p.exists():
        age_h = (time.time() - p.stat().st_mtime) / 3600
        if age_h < ttl_h:
            try:
                df = (pl.read_parquet(p).to_pandas() if _POLARS_OK
                      else pd.read_parquet(p))
                return _normalize_index(df)
            except Exception:
                pass  # corrupt file → re-fetch below

    # ── Cache miss / expired — fetch live ─────────────────────────────────
    sym = ticker.replace(".NS", "")
    raw = fetch_async([sym], period, interval, concurrency=1)
    df  = raw.get(sym, pd.DataFrame())
    if df.empty:
        try:
            import yfinance as yf
            df = yf.download(ticker, period=period, interval=interval,
                             auto_adjust=True, progress=False, threads=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
        except Exception:
            pass

    # ── Persist to disk ────────────────────────────────────────────────────
    if not df.empty and _PARQUET_OK:
        try:
            df_save = df.copy()
            if isinstance(df_save.index, pd.DatetimeIndex):
                df_save.index.name = "ts"
                df_save = df_save.reset_index()
            df_save.to_parquet(p, index=False, compression="snappy")
        except Exception:
            pass

    return df

def _htf_trend_from_df(df: pd.DataFrame, mode: str):
    if df is None or df.empty: return True, "HTF-UNKNOWN"
    if mode == "Intraday" and len(df) > 2: df = df.iloc[:-1].copy()
    min_bars = 55 if mode == "Intraday" else 26
    if len(df) < min_bars: return True, "HTF-UNKNOWN"
    cl = df["Close"]
    ef = float(ema(cl, 21 if mode == "Intraday" else 13).iloc[-1])
    es = float(ema(cl, 55 if mode == "Intraday" else 26).iloc[-1])
    c  = float(cl.iloc[-1])
    up = c > ef > es
    return up, ("HTF↑" if up else "HTF↓")

def prefetch_htf_parallel(symbols: list, mode: str, status_text, progress_bar) -> dict:
    """
    HTF pre-fetch with sampling — PERF fix for NSE-500 scale.

    At 500 survivors, fetching HTF for every stock is the #1 latency bottleneck.
    Strategy:
      • Sample up to _HTF_MAX symbols (spread uniformly so we cover high+low scores).
      • Default all others optimistically to (True, "HTF↑").
      • This cuts HTF round-trips by ~65% with minimal signal loss: HTF trend
        flips slowly (weekly / monthly bar) so the majority of stocks that were
        in uptrend last scan still are, and Stage-A already culled the broken ones.
    """
    cfg   = MODE_CFG[mode]
    total = len(symbols)

    # Tune: fetch HTF for at most _HTF_MAX stocks (spread over the full list)
    _HTF_MAX = 80
    if total <= _HTF_MAX:
        sample = symbols
    else:
        step   = total / _HTF_MAX
        sample = [symbols[int(i * step)] for i in range(_HTF_MAX)]

    # Default: optimistic HTF-up for all
    results: dict = {sym: (True, "HTF↑") for sym in symbols}

    if not sample:
        return results

    status_text.text(f"📡 HTF sample: {len(sample)}/{total} symbols…")
    raw = fetch_async(sample, cfg["htf_period"], cfg["htf_interval"], concurrency=20)
    for i, sym in enumerate(sample):
        df = raw.get(sym, pd.DataFrame())
        results[sym] = _htf_trend_from_df(df, mode)
        # v16.1: write to disk cache so individual score_stock calls get a hit
        if not df.empty and _PARQUET_OK:
            try:
                p = _htf_cache_path(sym, cfg["htf_interval"])
                df_save = df.copy()
                if isinstance(df_save.index, pd.DatetimeIndex):
                    df_save.index.name = "ts"
                    df_save = df_save.reset_index()
                df_save.to_parquet(p, index=False, compression="snappy")
            except Exception:
                pass
        if i % 20 == 0:
            progress_bar.progress(0.15 + i / max(total, 1) * 0.25)
    return results

# ══════════════════════════════════════════════════════════════════════════════
# RS RANKS (vectorized, unchanged)
# ══════════════════════════════════════════════════════════════════════════════


@st.cache_data(ttl=300)
def fetch_nifty(mode="Swing"):
    cfg = MODE_CFG[mode]
    raw = fetch_async(["^NSEI"], cfg["period"], cfg["interval"], concurrency=1)
    df  = raw.get("^NSEI", pd.DataFrame())
    if df.empty:
        try:
            import yfinance as yf
            df = yf.download("^NSEI", period=cfg["period"], interval=cfg["interval"], progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        except Exception:
            return pd.Series(dtype=float)
    return df["Close"].dropna()

def _market_regime(nifty_close):
    if len(nifty_close) < 50: return True, "UNKNOWN"
    ema20 = float(ema(nifty_close, 20).iloc[-1])
    ema50 = float(ema(nifty_close, 50).iloc[-1])
    bull  = (float(nifty_close.iloc[-1]) > ema50) and (ema20 > ema50)
    return bull, ("BULLISH" if bull else "BEARISH")

# ── v16.1: Precomputed regime cache — TTL 1 h, keyed by mode ──────────────────
# Regime is derived from Nifty EMAs which flip at most once per session.
# Caching here avoids re-fetching Nifty + re-computing EMAs on every scan.
_REGIME_CACHE_FILE = _CACHE_DIR / "regime_cache.json"

@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_regime(mode: str = "Swing") -> tuple:
    """
    Returns (market_bullish: bool, regime_label: str, nifty_last: float).
    Result is cached in memory for 1 h and also persisted to disk so the
    Supabase cron worker can pre-warm it before market open.
    """
    try:
        nifty = fetch_nifty(mode)
        bull, label = _market_regime(nifty)
        nifty_last  = float(nifty.iloc[-1]) if len(nifty) else 0.0
        payload = {"mode": mode, "bull": bull, "label": label,
                   "nifty_last": nifty_last,
                   "ts": datetime.utcnow().isoformat()}
        try:
            with open(_REGIME_CACHE_FILE, "w") as f:
                json.dump(payload, f)
        except Exception:
            pass
        return bull, label, nifty_last
    except Exception:
        # Fallback: try disk cache
        try:
            if _REGIME_CACHE_FILE.exists():
                with open(_REGIME_CACHE_FILE) as f:
                    p = json.load(f)
                if p.get("mode") == mode:
                    return p["bull"], p["label"], p.get("nifty_last", 0.0)
        except Exception:
            pass
        return True, "UNKNOWN", 0.0

# ══════════════════════════════════════════════════════════════════════════════
# SPEED-10: ADX + SQUEEZE helpers per-symbol (Stage-B enrichment)
# ══════════════════════════════════════════════════════════════════════════════


@st.cache_data(ttl=3600, show_spinner=False)
def get_earnings_dates(symbols: list) -> dict:
    """
    Returns {symbol: "DD Mon"} for stocks with earnings in the next 14 days.
    Best-effort — silent on any failure. Uses yfinance calendar.
    """
    import yfinance as yf
    from datetime import date as _date, timedelta as _td
    upcoming: dict = {}
    today   = _date.today()
    horizon = today + _td(days=14)
    for sym in symbols:
        try:
            cal = yf.Ticker(sym + ".NS").calendar
            if cal is None:
                continue
            if isinstance(cal, pd.DataFrame) and not cal.empty:
                if "Earnings Date" in cal.columns:
                    ed = pd.to_datetime(cal["Earnings Date"].iloc[0]).date()
                    if today <= ed <= horizon:
                        upcoming[sym] = ed.strftime("%d %b")
            elif isinstance(cal, dict):
                ed_raw = cal.get("Earnings Date")
                if ed_raw:
                    ed = (pd.to_datetime(ed_raw[0]).date()
                          if isinstance(ed_raw, list)
                          else pd.to_datetime(ed_raw).date())
                    if today <= ed <= horizon:
                        upcoming[sym] = ed.strftime("%d %b")
        except Exception:
            pass
    return upcoming

# ══════════════════════════════════════════════════════════════════════════════
# OI DATA — improved NSE session warm-up (v15.8-FIX)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=180)
def fetch_oi_data(symbol="NIFTY"):
    import requests
    HEADERS={
        "User-Agent":("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"),
        "Accept":"application/json, text/plain, */*",
        "Accept-Language":"en-US,en;q=0.9",
        "Accept-Encoding":"gzip, deflate, br",
        "Referer":"https://www.nseindia.com/",
        "X-Requested-With":"XMLHttpRequest","Connection":"keep-alive",
        "Cache-Control":"no-cache",
        "Sec-Fetch-Site":"same-origin","Sec-Fetch-Mode":"cors","Sec-Fetch-Dest":"empty",
    }
    session=requests.Session(); session.headers.update(HEADERS)
    def _warm():
        # v15.8-FIX: proper NSE session warming — needs gap between requests for cookie setup
        try:
            session.get("https://www.nseindia.com", timeout=8,
                        headers={**HEADERS,
                                  "Accept":"text/html,application/xhtml+xml,*/*;q=0.8",
                                  "Sec-Fetch-Mode":"navigate","Sec-Fetch-Dest":"document"})
            time.sleep(1.5)   # NSE needs this gap to set session cookies
            session.get("https://www.nseindia.com/market-data/equity-derivatives-watch", timeout=8)
            time.sleep(1.0)
            return "nsit" in session.cookies or "nseappid" in session.cookies
        except Exception:
            return False
    _warm()
    oc_url=f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    data=None
    for attempt in range(3):
        try:
            resp=session.get(oc_url,timeout=12)
            if resp.status_code==200: data=resp.json(); break
            elif resp.status_code in (401,403): _warm()
        except Exception: pass
        time.sleep(1.5**attempt)
    if data is None: return None
    try:
        records=data["records"]; spot=float(records["underlyingValue"])
        expiries=records["expiryDates"]; weekly_expiry=expiries[0] if expiries else None
        rows=[]
        for item in records["data"]:
            if item.get("expiryDate")!=weekly_expiry: continue
            strike=item["strikePrice"]
            ce_oi=item.get("CE",{}).get("openInterest",0) or 0
            pe_oi=item.get("PE",{}).get("openInterest",0) or 0
            ce_chg=item.get("CE",{}).get("changeinOpenInterest",0) or 0
            pe_chg=item.get("PE",{}).get("changeinOpenInterest",0) or 0
            rows.append({"Strike":strike,"CE_OI":ce_oi,"CE_Chg":ce_chg,"PE_OI":pe_oi,"PE_Chg":pe_chg})
        if not rows: return None
        df_oi=pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)
        total_ce=df_oi["CE_OI"].sum(); total_pe=df_oi["PE_OI"].sum()
        pcr=round(total_pe/total_ce,2) if total_ce>0 else 0
        pains=[]
        for s in df_oi["Strike"]:
            ce_l=((df_oi["Strike"]-s).clip(lower=0)*df_oi["CE_OI"]).sum()
            pe_l=((s-df_oi["Strike"]).clip(lower=0)*df_oi["PE_OI"]).sum()
            pains.append(ce_l+pe_l)
        df_oi["TotalPain"]=pains
        return {
            "symbol":symbol,"expiry":weekly_expiry,"spot":spot,"pcr":pcr,
            "max_pain":int(df_oi.loc[df_oi["TotalPain"].idxmin(),"Strike"]),
            "call_wall":int(df_oi.loc[df_oi["CE_OI"].idxmax(),"Strike"]),
            "put_wall":int(df_oi.loc[df_oi["PE_OI"].idxmax(),"Strike"]),
            "top_ce":df_oi.nlargest(5,"CE_OI")[["Strike","CE_OI","CE_Chg"]].to_dict("records"),
            "top_pe":df_oi.nlargest(5,"PE_OI")[["Strike","PE_OI","PE_Chg"]].to_dict("records"),
            "df_oi":df_oi,
        }
    except Exception: return None

def _oi_sentiment(pcr):
    if pcr>=1.3: return "Bullish","#16a34a"
    if pcr>=0.9: return "Neutral","#d97706"
    return "Bearish","#dc2626"

@st.cache_data(ttl=300)
def fetch_indices(mode="Swing"):
    cfg=MODE_CFG[mode]; ema_f=cfg["ema_fast"]; ema_s=cfg["ema_slow"]; rsi_l=cfg["rsi_len"]
    min_bars=60 if mode=="Intraday" else 50; out={}
    index_syms=[("Nifty 50","^NSEI"),("BankNifty","^NSEBANK"),("Sensex","^BSESN")]
    raw=fetch_async(["^NSEI","^NSEBANK","^BSESN"],cfg["yf_period"],cfg["interval"],concurrency=3)
    for name,ticker in index_syms:
        sym_key=ticker
        df=raw.get(sym_key,pd.DataFrame())
        if df.empty:
            out[name]=None; continue
        try:
            if len(df)<min_bars: out[name]=None; continue
            close=df["Close"]; c,prev=float(close.iloc[-1]),float(close.iloc[-2])
            chg,pct=c-prev,(c-prev)/prev*100
            ef=float(ema(close,ema_f).iloc[-1]); es=float(ema(close,ema_s).iloc[-1])
            e200=float(ema(close,200).iloc[-1]) if len(close)>=200 else es
            r=float(rsi(close,rsi_l).iloc[-1]); hh=float(close.iloc[-11:-1].max())
            trend_up=c>e200 and c>ef and ef>es
            bull=0
            bull+=25 if trend_up else 0
            bull+=15 if ef>es else (7 if ef>es*0.995 else 0)
            bull+=(15 if r>=65 else 10) if r>=60 else (5 if r>50 else 0)
            bull+=15 if c>hh else (9 if c>hh*0.98 else 0)
            if len(close)>=3 and c>float(close.iloc[-3]): bull+=8
            norm_score=min(100.0,max(0.0,bull*100.0/78))
            interval_label={"5m":"5min","1d":"Daily","1wk":"Weekly"}.get(cfg["interval"],cfg["interval"])
            out[name]={"value":round(c,1),"chg":round(chg,2),"pct":round(pct,2),
                       "score":round(norm_score,1),"action":action_label(norm_score),
                       "rsi":round(r,1),"trend":"↑ Above EMAs" if trend_up else "↓ Below EMAs",
                       "interval":interval_label,"ema_fast":ema_f,"ema_slow":ema_s}
        except Exception:
            out[name]=None
    return out

# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🐂 BULL SUTRA Pro v16.0",
    page_icon="🐂", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@400;500;600&family=Syne:wght@600;700&display=swap');
html,body,[class*="css"]{background:#07070f;color:#e8e8f4;}
.stApp{background:#07070f;}
.stDataFrame{background:#111120;}
.stButton>button{background:#1a1a35;border:1px solid #2a2a55;color:#e8e8f4;border-radius:8px;}
.stButton>button[kind="primary"]{background:#f59e0b;color:#1a0a00;font-weight:700;}
[data-testid="metric-container"]{background:#111120;border:1px solid #1e1e40;border-radius:8px;padding:10px;}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for key,default in [
    ("results",[]),("scan_time",None),("rejected",0),("liq_skipped",0),
    ("scan_mode","Swing"),("signal_log",[]),("phase_history",{}),
    ("account_size",500000),("risk_pct",0.02),("max_capital_pct",0.20),
    ("phase_filter","All Phases"),("show_illiquid",False),("min_liq_cr",5.0),
    ("open_positions",None),("short_results",[]),("short_watchlist",None),
    ("exit_results",{}),("_db_error",None),
    # v15 additions
    ("last_scan_stage_a_survivors",0),("live_refresh_enabled",False),
    ("earnings_map",{}),  # v15.8-FIX: earnings date cache
    # v16.1: Top5 persistence + last-scan memory
    ("top5",[]),                   # pre-computed at scan time, never re-derived at render
    ("last_scan_meta", None),      # {universe, mode, elapsed, ts, n_scored, regime}
    ("last_scan_results_cache", None),  # disk-backed scan result cache path
    # v16.1: lazy UI — track which secondary tabs have been loaded at least once
    ("tab_sectors_loaded",  False),
    ("tab_breadth_loaded",  False),
    ("tab_analytics_loaded",False),
    ("tab_detail_loaded",   False),
    ("enrichment_ready",    True),   # True = no enrichment pending
]:
    if key not in st.session_state:
        st.session_state[key]=default

if st.session_state["open_positions"] is None:
    st.session_state["open_positions"]=_db_load("bs_positions")
if st.session_state["short_watchlist"] is None:
    st.session_state["short_watchlist"]=_db_load("bs_short_wl")

# ── v16.1: Restore last scan from disk cache on page refresh ──────────────────
_SCAN_CACHE_FILE = _CACHE_DIR / "last_scan_results.json"
_SCAN_META_FILE  = _CACHE_DIR / "last_scan_meta.json"

