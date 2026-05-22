"""
config.py — All shared constants, mode config, and environment globals.
"""
import warnings
import logging
import time
import os
import threading
import concurrent.futures
import asyncio
import json
import hashlib
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ── Optional fast-path imports ─────────────────────────────────────────────────
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
    import yfinance as yf          # fallback

try:
    import pyarrow                 # needed for parquet cache
    _PARQUET_OK = True
except ImportError:
    _PARQUET_OK = False

try:
    import psycopg2 as _psycopg2
    _DB_OK = True
except ImportError:
    _DB_OK = False

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

# ── Cache directory ────────────────────────────────────────────────────────────
_CACHE_DIR = Path(os.environ.get("BS_CACHE_DIR", "/tmp/bull_sutra_cache"))
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Universes ──────────────────────────────────────────────────────────────────
try:
    from sectors import SECTORS as _SECTORS
except ImportError:
    _SECTORS = None
    try:
        import urllib.request, types as _types, hashlib as _hashlib
        _GH_SECTORS_URL = (
            "https://raw.githubusercontent.com/srikanthgodavarthy/nse-scan/main/sectors.py"
        )
        # SECURITY NOTE: exec() on remote code is a risk. Pin to a known SHA-256 or
        # bundle sectors.py locally. The hash check below should be updated whenever
        # sectors.py changes on the remote. Set _SECTORS_EXPECTED_SHA = None to skip.
        _SECTORS_EXPECTED_SHA = None   # e.g. "abc123..." — set to your known hash
        with urllib.request.urlopen(_GH_SECTORS_URL, timeout=10) as _resp:
            _src_bytes = _resp.read()
        if _SECTORS_EXPECTED_SHA is not None:
            _actual_sha = _hashlib.sha256(_src_bytes).hexdigest()
            if _actual_sha != _SECTORS_EXPECTED_SHA:
                raise RuntimeError(
                    f"sectors.py integrity check failed: got {_actual_sha[:16]}…"
                )
        _src = _src_bytes.decode("utf-8")
        _mod = _types.ModuleType("sectors_remote")
        exec(compile(_src, "<sectors_gh>", "exec"), _mod.__dict__)
        _SECTORS = getattr(_mod, "SECTORS", None)
    except Exception:
        _SECTORS = None

try:
    from nse500 import nse500_symbols
    NSE500 = list(dict.fromkeys([s.strip().upper().replace(".NS","") for s in nse500_symbols]))
except ImportError:
    NSE500 = [
        "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC","SBIN",
        "BHARTIARTL","KOTAKBANK","LT","AXISBANK","ASIANPAINT","MARUTI","TITAN",
        "NESTLEIND","WIPRO","ULTRACEMCO","POWERGRID","NTPC","BAJFINANCE","HCLTECH",
        "SUNPHARMA","TECHM","INDUSINDBK","ONGC","COALINDIA","TATASTEEL","JSWSTEEL",
        "HINDALCO","TATAMOTORS","M&M","BAJAJFINSV","DIVISLAB","DRREDDY","CIPLA",
        "EICHERMOT","ADANIENT","ADANIPORTS","BPCL","TATACONSUM","BRITANNIA",
        "HEROMOTOCO","APOLLOHOSP","GRASIM","SBILIFE","HDFCLIFE","ICICIPRULI","VEDL","NMDC",
    ]

NIFTY50 = (
    _SECTORS["Nifty 50"]
    if _SECTORS and "Nifty 50" in _SECTORS
    else [
        "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC","SBIN",
        "BHARTIARTL","KOTAKBANK","LT","AXISBANK","ASIANPAINT","MARUTI","TITAN",
        "NESTLEIND","WIPRO","ULTRACEMCO","POWERGRID","NTPC","BAJFINANCE","HCLTECH",
        "SUNPHARMA","TECHM","INDUSINDBK","ONGC","COALINDIA","TATASTEEL","JSWSTEEL",
        "HINDALCO","TATAMOTORS","M&M","BAJAJFINSV","DIVISLAB","DRREDDY","CIPLA",
        "EICHERMOT","ADANIENT","ADANIPORTS","BPCL","TATACONSUM","BRITANNIA",
        "HEROMOTOCO","APOLLOHOSP","GRASIM","SBILIFE","HDFCLIFE","ICICIPRULI","BAJAJ-AUTO","UPL",
    ]
)

# ── SECTOR_MAP ─────────────────────────────────────────────────────────────────
SECTOR_MAP: dict[str, str] = {}

_CSV_GH_URL = (
    "https://raw.githubusercontent.com/srikanthgodavarthy/nse-scan/main/nse500_clean_sample.csv"
)

def _load_sector_csv(source) -> dict:
    _df = pd.read_csv(source)
    _df["Symbol"] = _df["Symbol"].astype(str).str.replace(".NS","",regex=False).str.strip()
    _sector_col = "Sector" if "Sector" in _df.columns else "Industry"
    _df[_sector_col] = _df[_sector_col].astype(str).str.strip().str.title()
    return dict(zip(_df["Symbol"], _df[_sector_col]))

try:
    SECTOR_MAP = _load_sector_csv(_CSV_GH_URL)
except Exception:
    try:
        SECTOR_MAP = _load_sector_csv("nse500_clean_sample.csv")
    except Exception:
        pass

if not SECTOR_MAP:
    if _SECTORS:
        for _sector_name, _syms in _SECTORS.items():
            if _syms is None:
                continue
            for _sym in _syms:
                if _sym not in SECTOR_MAP:
                    SECTOR_MAP[_sym] = _sector_name
    if not SECTOR_MAP:
        SECTOR_MAP = {
            "RELIANCE":"Energy & Power","ONGC":"Energy & Power","BPCL":"Energy & Power",
            "COALINDIA":"Energy & Power","NTPC":"Energy & Power","POWERGRID":"Energy & Power",
            "ADANIENT":"Energy & Power","ADANIPORTS":"Infrastructure","LT":"Infrastructure",
            "HDFCBANK":"Banking & Finance","ICICIBANK":"Banking & Finance",
            "SBIN":"Banking & Finance","KOTAKBANK":"Banking & Finance",
            "AXISBANK":"Banking & Finance","BAJFINANCE":"Banking & Finance",
            "TCS":"IT & Technology","INFY":"IT & Technology","WIPRO":"IT & Technology",
            "HCLTECH":"IT & Technology","TECHM":"IT & Technology",
            "SUNPHARMA":"Pharma & Healthcare","DRREDDY":"Pharma & Healthcare",
            "CIPLA":"Pharma & Healthcare","HINDUNILVR":"FMCG & Consumer",
            "ITC":"FMCG & Consumer","NESTLEIND":"FMCG & Consumer",
            "TATASTEEL":"Metals & Mining","JSWSTEEL":"Metals & Mining",
            "MARUTI":"Auto & Auto Ancillaries","TATAMOTORS":"Auto & Auto Ancillaries",
        }

# ── Secondary sector lookup from _SECTORS (covers stocks not in the CSV) ──────
# Many symbols fall into "Other" if the GitHub CSV download fails or the CSV
# doesn't include them. This parallel dict uses _SECTORS (sectors.py) as a
# clean-name fallback and is merged at lookup time in run_scan.
_SECTORS_LOOKUP: dict[str, str] = {}
if _SECTORS:
    _SKIP_META = {"Nifty 50", "Nifty 500", "NSE 500"}   # index names, not sectors
    for _sname, _syms in _SECTORS.items():
        if _syms is None or _sname in _SKIP_META:
            continue
        for _sym in (_syms if isinstance(_syms, list) else []):
            _clean = str(_sym).strip().upper().replace(".NS", "")
            if _clean and _clean not in _SECTORS_LOOKUP:
                _SECTORS_LOOKUP[_clean] = _sname

# ── Mode config ────────────────────────────────────────────────────────────────
MODE_CFG = {
    "Intraday":   dict(period="5d",  interval="5m",  ema_fast=9,  ema_slow=21,
                       atr_mult=1.5, atr_wide=3.0, atr_max=1.0,
                       mom1_th=2, mom3_th=5, mom6_th=8, score_th=65, rsi_len=14,
                       htf_period="3mo", htf_interval="15m", validity_hours=4,
                       yf_period="5d",  yf_interval="5m",
                       live_interval="1m", hist_min_bars=60),
    "Swing":      dict(period="1y",  interval="1d",  ema_fast=50, ema_slow=200,
                       atr_mult=2.5, atr_wide=4.0, atr_max=1.5,
                       mom1_th=3, mom3_th=7, mom6_th=10, score_th=70, rsi_len=21,
                       htf_period="2y", htf_interval="1wk", validity_hours=72,
                       yf_period="1y",  yf_interval="1d",
                       live_interval="1d", hist_min_bars=50),
    "Positional": dict(period="2y",  interval="1d",  ema_fast=50, ema_slow=200,
                       atr_mult=3.5, atr_wide=5.0, atr_max=1.5,
                       mom1_th=5, mom3_th=10, mom6_th=15, score_th=70, rsi_len=21,
                       htf_period="5y", htf_interval="1wk", validity_hours=240,
                       yf_period="2y",  yf_interval="1d",
                       live_interval="1d", hist_min_bars=50),
}

BULL_MAX        = 120
ACTION_THRESHOLDS = dict(strong_buy=75, buy=58, watch=42)

PHASE_IDLE  = "IDLE";  PHASE_SETUP = "SETUP"; PHASE_ENTRY = "ENTRY"
PHASE_CONT  = "CONT";  PHASE_BRK   = "BREAKOUT"; PHASE_EXIT = "EXIT"

PHASE_COLORS = {
    PHASE_IDLE:"#555577", PHASE_SETUP:"#b87333",
    PHASE_ENTRY:"#2255cc", PHASE_CONT:"#22aa55",
    PHASE_BRK:"#00dd88",  PHASE_EXIT:"#cc4444",
}
PHASE_ORDER = {
    PHASE_IDLE:0, PHASE_SETUP:1, PHASE_ENTRY:2,
    PHASE_CONT:3, PHASE_BRK:4, PHASE_EXIT:-1,
}

VIX_CALM=15; VIX_CAUTION=20; VIX_STRESS=20  # v15.8-FIX: was 25 — UI said STRESS at 20 but math used 25
LIQUIDITY_MIN_CR = 5.0

EXIT_HOLD="HOLD"; EXIT_WATCH_LBL="EXIT WATCH"
EXIT_SIGNAL_LBL="EXIT SIGNAL"; EXIT_CONFIRM_LBL="EXIT NOW"
EXIT_COLORS = {
    EXIT_HOLD:"#22aa55", EXIT_WATCH_LBL:"#f59e0b",
    EXIT_SIGNAL_LBL:"#ff8800", EXIT_CONFIRM_LBL:"#cc4444",
}

SHORT_SKIP="SKIP"; SHORT_WATCH="SHORT WATCH"
SHORT_SIGNAL="SHORT SIGNAL"; SHORT_CONFIRMED="SHORT NOW"
SHORT_COLORS = {
    SHORT_SKIP:"#555577", SHORT_WATCH:"#f59e0b",
    SHORT_SIGNAL:"#ff6b35", SHORT_CONFIRMED:"#cc2244",
}
SHORT_SCORE_WATCH=25; SHORT_SCORE_SIGNAL=45
SHORT_SCORE_CONFIRMED=68; SHORT_HARD_WEIGHT=22; SHORT_SOFT_WEIGHT=9

NSE_OPEN_HOUR=9;  NSE_OPEN_MIN=15
NSE_CLOSE_HOUR=15; NSE_CLOSE_MIN=30
NSE_SESSION_MINUTES = (NSE_CLOSE_HOUR*60+NSE_CLOSE_MIN)-(NSE_OPEN_HOUR*60+NSE_OPEN_MIN)

_phase_lock = threading.Lock()

# ── Regime constants (used by scoring.py and market.py) ──────────────────────
REGIME_TREND        = "TREND"
REGIME_ROTATION     = "ROTATION"
REGIME_DISTRIBUTION = "DISTRIBUTION"
REGIME_PANIC        = "PANIC"
REGIME_EXPANSION    = "EXPANSION"

_REGIME_ADJUSTMENTS = {
    REGIME_EXPANSION:    {"score_floor": 45, "target_mult": 1.20, "preconfirm": "aggressive", "sl_mult": 1.0,  "size_pct": 1.00},
    REGIME_TREND:        {"score_floor": 50, "target_mult": 1.00, "preconfirm": "normal",     "sl_mult": 1.0,  "size_pct": 1.00},
    REGIME_ROTATION:     {"score_floor": 55, "target_mult": 0.90, "preconfirm": "selective",  "sl_mult": 1.1,  "size_pct": 0.75},
    REGIME_DISTRIBUTION: {"score_floor": 65, "target_mult": 0.75, "preconfirm": "off",        "sl_mult": 1.2,  "size_pct": 0.50},
    REGIME_PANIC:        {"score_floor": 80, "target_mult": 0.60, "preconfirm": "off",        "sl_mult": 1.5,  "size_pct": 0.25},
}

# ── Phase constants missing from config ──────────────────────────────────────
PHASE_SETUP = "SETUP"
PHASE_ENTRY = "ENTRY"
PHASE_BRK   = "BREAKOUT"
PHASE_EXIT  = "EXIT"

# ── VIX constants ─────────────────────────────────────────────────────────────
VIX_CAUTION = 20
VIX_STRESS  = 20

# ── Exit label constants ──────────────────────────────────────────────────────
EXIT_WATCH_LBL   = "EXIT WATCH"
EXIT_SIGNAL_LBL  = "EXIT SIGNAL"
EXIT_CONFIRM_LBL = "EXIT NOW"