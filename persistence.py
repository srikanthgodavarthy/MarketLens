"""
persistence.py — Supabase/PostgreSQL state, earnings cache, scan-result disk cache.
"""
import json
import time
import logging
import hashlib
import struct
import streamlit as st
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import psycopg2 as _psycopg2
    _DB_OK = True
except ImportError:
    _DB_OK = False

from config import _CACHE_DIR, LIQUIDITY_MIN_CR
from market import compute_breadth
from data_fetch import get_earnings_dates

_SCAN_CACHE_FILE = _CACHE_DIR / "last_scan_results.json"
_SCAN_META_FILE  = _CACHE_DIR / "last_scan_meta.json"

def _db_conn():
    if not _DB_OK: raise RuntimeError("psycopg2 not installed")
    url = st.secrets.get("SUPABASE_URL","")
    if not url or not url.startswith(("postgres://","postgresql://")):
        raise ValueError("SUPABASE_URL missing/malformed")
    return _psycopg2.connect(url)

def _db_ensure(cur):
    cur.execute("""CREATE TABLE IF NOT EXISTS bs_positions (
        id SERIAL PRIMARY KEY, data JSONB NOT NULL, ts TIMESTAMP DEFAULT now())""")
    cur.execute("""CREATE TABLE IF NOT EXISTS bs_short_wl (
        id SERIAL PRIMARY KEY, data JSONB NOT NULL, ts TIMESTAMP DEFAULT now())""")

def _db_save(table, payload):
    try:
        conn = _db_conn(); cur = conn.cursor()
        _db_ensure(cur); conn.commit()
        # FIX-6: atomic save — insert first, then trim old rows in one transaction
        # This prevents data loss if the process crashes between a DELETE and INSERT.
        cur.execute(
            f"INSERT INTO {table} (data) VALUES (%s)",
            [json.dumps(payload)]
        )
        cur.execute(
            f"""DELETE FROM {table}
                WHERE id NOT IN (
                    SELECT id FROM {table} ORDER BY ts DESC LIMIT 1
                )"""
        )
        conn.commit(); cur.close(); conn.close()
        st.session_state["_db_error"] = None
    except Exception as e:
        st.session_state["_db_error"] = str(e)

def _db_load(table):
    try:
        conn=_db_conn(); cur=conn.cursor()
        _db_ensure(cur); conn.commit()
        cur.execute(f"SELECT data FROM {table} ORDER BY ts DESC LIMIT 1")
        row=cur.fetchone(); cur.close(); conn.close()
        if row and row[0]:
            return row[0] if isinstance(row[0],list) else json.loads(row[0])
    except Exception:
        pass
    return []

# ══════════════════════════════════════════════════════════════════════════════
# v16.1: SUPABASE WORKER TABLES + LAZY EARNINGS
# Heavy jobs (earnings, regime, HTF batch) are offloaded to a Supabase cron.
# The Streamlit app reads from these tables and falls back to local compute
# only on a miss.  pg_cron SQL (run once in Supabase SQL Editor):
#
#   SELECT cron.schedule('bs-regime',  '*/30 * * * *',
#       $$SELECT net.http_post(url:='<YOUR_EDGE_FN_URL>/regime')$$);
#   SELECT cron.schedule('bs-earnings','0 8 * * 1-5',
#       $$SELECT net.http_post(url:='<YOUR_EDGE_FN_URL>/earnings')$$);
# ══════════════════════════════════════════════════════════════════════════════

def _db_ensure_worker_tables(cur):
    """Create worker cache tables if they don't exist (idempotent)."""
    cur.execute("""CREATE TABLE IF NOT EXISTS bs_earnings_cache (
        id SERIAL PRIMARY KEY,
        data JSONB NOT NULL,
        computed_for_date DATE DEFAULT CURRENT_DATE,
        ts TIMESTAMP DEFAULT now())""")
    cur.execute("""CREATE TABLE IF NOT EXISTS bs_regime_cache (
        id SERIAL PRIMARY KEY,
        mode TEXT NOT NULL,
        data JSONB NOT NULL,
        ts TIMESTAMP DEFAULT now())""")

def _supabase_load_earnings() -> dict:
    """Read earnings from Supabase cache (written by cron worker). TTL: 12 h."""
    try:
        conn = _db_conn(); cur = conn.cursor()
        _db_ensure(cur); _db_ensure_worker_tables(cur); conn.commit()
        cur.execute("""SELECT data, ts FROM bs_earnings_cache
                       ORDER BY ts DESC LIMIT 1""")
        row = cur.fetchone(); cur.close(); conn.close()
        if row and row[0]:
            age_h = (datetime.utcnow() - row[1]).total_seconds() / 3600
            if age_h < 12:
                return json.loads(row[0]) if isinstance(row[0], str) else row[0]
    except Exception:
        pass
    return {}

def _supabase_save_earnings(earnings_map: dict):
    """Persist earnings map to Supabase for cron worker reuse."""
    try:
        conn = _db_conn(); cur = conn.cursor()
        _db_ensure(cur); _db_ensure_worker_tables(cur); conn.commit()
        cur.execute("INSERT INTO bs_earnings_cache (data) VALUES (%s)",
                    [json.dumps(earnings_map)])
        cur.execute("""DELETE FROM bs_earnings_cache
                       WHERE id NOT IN (
                           SELECT id FROM bs_earnings_cache ORDER BY ts DESC LIMIT 3
                       )""")
        conn.commit(); cur.close(); conn.close()
    except Exception:
        pass

@st.cache_data(ttl=3600, show_spinner=False)
def _get_earnings_cached(symbols_key: str, symbols: tuple) -> dict:
    """
    Lazy earnings loader — called only when Scanner tab renders, not during scan.
    Priority: (1) Supabase cache → (2) local yfinance fetch → (3) empty dict.
    Result cached in memory for 1 h so tab switches don't re-fetch.
    """
    # Try Supabase first
    em = _supabase_load_earnings()
    if em:
        return em
    # Fall back to local fetch (sequential — acceptable since it's lazy)
    em = get_earnings_dates(list(symbols))
    if em:
        _supabase_save_earnings(em)
    return em

# ══════════════════════════════════════════════════════════════════════════════
# RUN SCAN — v15: two-stage + incremental + async HTTP
# ══════════════════════════════════════════════════════════════════════════════


def _result_hash(r: dict) -> str:
    """Stable short hash of a result dict for Streamlit key uniqueness."""
    key_fields = (r.get("Symbol",""), r.get("Score",0), r.get("Phase",""),
                  r.get("Action",""), r.get("LTP",0), r.get("Confidence",0))
    return hashlib.md5(str(key_fields).encode()).hexdigest()[:8]

# ══════════════════════════════════════════════════════════════════════════════
# v15.8-FIX: EARNINGS DATE WARNING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)

def _save_scan_cache(results: list, meta: dict):
    """Persist scan results + meta to disk so they survive page refresh."""
    try:
        with open(_SCAN_CACHE_FILE, "w") as f:
            json.dump(results, f, default=str)
        with open(_SCAN_META_FILE, "w") as f:
            json.dump(meta, f, default=str)
    except Exception:
        pass

def _load_scan_cache() -> tuple:
    """Load scan results + meta from disk. Returns (results, meta) or ([], None)."""
    try:
        if _SCAN_CACHE_FILE.exists() and _SCAN_META_FILE.exists():
            age_hours = (time.time() - _SCAN_CACHE_FILE.stat().st_mtime) / 3600
            if age_hours < 12:                     # discard cache older than 12 h
                with open(_SCAN_CACHE_FILE) as f:
                    results = json.load(f)
                with open(_SCAN_META_FILE) as f:
                    meta = json.load(f)
                return results, meta
    except Exception:
        pass
    return [], None

def _compute_top5(results: list) -> list:
    """Derive Top-5 from results list — single source of truth."""
    candidates = [
        r for r in results
        if r.get("LiquidityOK", True)
        and r.get("ExtN", 0) <= 1
        and r.get("ReadinessScore", 0) >= 55
    ]
    candidates.sort(
        key=lambda x: (x.get("ReadinessScore", 0), x.get("RSLeaderScore", 0)),
        reverse=True,
    )
    return candidates[:5]

# Restore results from disk if session_state is empty (e.g. after page refresh)
if not st.session_state["results"] and not st.session_state["last_scan_meta"]:
    _cached_results, _cached_meta = _load_scan_cache()
    if _cached_results:
        st.session_state["results"]         = _cached_results
        st.session_state["last_scan_meta"]  = _cached_meta
        st.session_state["top5"]            = _compute_top5(_cached_results)
        st.session_state["breadth"]         = compute_breadth(_cached_results)
        if _cached_meta:
            st.session_state["scan_mode"]   = _cached_meta.get("mode", "Swing")

# ── v16.1: Pick up background pattern enrichment when ready ───────────────────
_ENRICH_CACHE = _CACHE_DIR / "enrichment_pending.json"
if (not st.session_state.get("enrichment_ready", True)
        and _ENRICH_CACHE.exists()):
    try:
        _enrich_age = time.time() - _ENRICH_CACHE.stat().st_mtime
        if _enrich_age < 600:               # ignore files older than 10 min
            with open(_ENRICH_CACHE) as _ef:
                _enriched = json.load(_ef)
            if _enriched:
                st.session_state["results"] = _enriched
                st.session_state["top5"]    = _compute_top5(_enriched)
                st.session_state["enrichment_ready"] = True
                _ENRICH_CACHE.unlink(missing_ok=True)
    except Exception:
        pass

