"""
market.py — Market-level regime classification, breadth engine, RS ranks.
"""
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Optional

from config import (
    MODE_CFG, REGIME_TREND, REGIME_ROTATION, REGIME_DISTRIBUTION,
    REGIME_PANIC, REGIME_EXPANSION, _REGIME_ADJUSTMENTS,
    VIX_CALM, VIX_CAUTION, VIX_STRESS,
    PHASE_IDLE, PHASE_SETUP, PHASE_ENTRY, PHASE_CONT, PHASE_BRK, _CACHE_DIR,
)
from indicators import ema, rsi
from data_fetch import fetch_nifty, fetch_async, fetch_vix, _market_regime, get_cached_regime

def compute_market_regime(
    vix_val: float,
    pct_above_ema50: float,
    ad_ratio: float,
    pct_advancing: float,
    pct_breakout: float,
    nifty_close: pd.Series = None,
) -> dict:
    """
    Master regime engine — classifies market structure into one of five regimes.

    Inputs:
      vix_val          — India VIX (or ^VIX proxy)
      pct_above_ema50  — % of scanned stocks above their 50 EMA
      ad_ratio         — advancing / declining ratio
      pct_advancing    — % of stocks advancing today
      pct_breakout     — % of stocks in breakout phase
      nifty_close      — Nifty 50 close series for trend structure

    Returns dict with:
      regime       — one of the five REGIME_* constants
      regime_label — plain-English one-liner
      adjustments  — score floor, target mult, preconfirm mode, SL mult, size %
    """
    vix  = vix_val  or 15.0
    em50 = pct_above_ema50 or 50.0
    adr  = ad_ratio or 1.0
    padv = pct_advancing or 50.0
    pbrk = pct_breakout or 2.0

    # ── Nifty structure: is it in an uptrend? ─────────────────────────────
    nifty_up = True
    if nifty_close is not None and len(nifty_close) >= 50:
        try:
            ef = float(nifty_close.ewm(span=20, adjust=False).mean().iloc[-1])
            es = float(nifty_close.ewm(span=50, adjust=False).mean().iloc[-1])
            cl = float(nifty_close.iloc[-1])
            nifty_up = (cl > ef) and (ef > es)
        except Exception:
            nifty_up = True

    # ── Score each regime dimension ────────────────────────────────────────
    panic_score = (
        (2 if vix >= 25 else 1 if vix >= 20 else 0) +
        (2 if em50 < 25 else 1 if em50 < 35 else 0) +
        (2 if adr  < 0.5 else 1 if adr < 0.7 else 0)
    )
    dist_score = (
        (2 if vix >= 18 else 1 if vix >= 15 else 0) +
        (2 if em50 < 40 else 1 if em50 < 50 else 0) +
        (1 if adr  < 0.9 else 0) +
        (1 if not nifty_up else 0)
    )
    expan_score = (
        (2 if em50 >= 75 else 1 if em50 >= 65 else 0) +
        (2 if adr  >= 2.5 else 1 if adr >= 2.0 else 0) +
        (1 if pbrk >= 8 else 0) +
        (1 if nifty_up else 0)
    )
    rotation_score = (
        (1 if 50 <= em50 < 65 else 0) +
        (1 if 0.9 <= adr < 1.5 else 0) +
        (1 if pbrk < 4 else 0)
    )

    # ── Classify ───────────────────────────────────────────────────────────
    if panic_score >= 5:
        regime = REGIME_PANIC
        label  = "PANIC — breadth collapsed, VIX elevated. No long entries."
    elif dist_score >= 4:
        regime = REGIME_DISTRIBUTION
        label  = "DISTRIBUTION — breadth weakening. Size down 50%, exits only."
    elif expan_score >= 4:
        regime = REGIME_EXPANSION
        label  = "EXPANSION — broad advance, multiple breakouts. Full aggression."
    elif rotation_score >= 2:
        regime = REGIME_ROTATION
        label  = "ROTATION — sectors churning. Favour accumulation, reduce breakouts."
    else:
        regime = REGIME_TREND
        label  = "TREND — uptrend intact. Normal position sizing."

    return {
        "regime":       regime,
        "regime_label": label,
        "adjustments":  _REGIME_ADJUSTMENTS[regime],
    }


# ══════════════════════════════════════════════════════════════════════════════
# ORTHOGONALITY PENALTY
# Problem (Image 1): all engines rise together for a genuinely strong stock.
# This creates score stacking — ReadinessScore becomes overconfident.
#
# Fix: measure how correlated the sub-scores are. When PCA, SmartMoney,
# EmScore, and MicroScore all peak simultaneously, add a small penalty to
# ReadinessScore because the number is being inflated by correlation, not
# by independent evidence.
#
# The penalty is SMALL (max -8 points) — we don't want to punish truly
# great setups, just prevent 95/100 scores on everything.
#
# Penalty logic:
#   n_above_75 = count of sub-scores above 75
#   if n_above_75 >= 4: penalty = 8  (all four firing = stacking risk)
#   if n_above_75 == 3: penalty = 4
#   if n_above_75 <= 2: penalty = 0  (genuine independence — no penalty)
# ══════════════════════════════════════════════════════════════════════════════

def compute_orthogonality_penalty(
    pca_score: float,
    smart_money_score: float,
    em_score: float,
    micro_score: float,
    rs_score: float = 50.0,
) -> float:
    """
    Returns a score penalty (0–8) for correlated sub-score stacking.
    Higher penalty = more engines are simultaneously peaking = inflated confidence.
    """
    sub_scores = [pca_score, smart_money_score, em_score, micro_score]
    n_above_75 = sum(1 for s in sub_scores if s >= 75)
    n_above_85 = sum(1 for s in sub_scores if s >= 85)

    if n_above_85 >= 3:
        return 8.0   # three engines all very high — severe stacking
    if n_above_75 >= 4:
        return 6.0   # all four high — moderate stacking
    if n_above_75 >= 3:
        return 3.0   # three high — mild stacking
    return 0.0       # two or fewer — acceptable


# ══════════════════════════════════════════════════════════════════════════════
# OUTCOME FEEDBACK LOOP
# Problem (Image 4): every setup is evaluated independently. Failed setups
# are not remembered. Professionals track recent signal failures.
#
# Implementation: a lightweight in-session failure registry.
# When a setup is manually marked as failed (via portfolio), the symbol +
# setup_type + sector are recorded. These reduce confidence for similar
# setups in the next scan within the same session.
#
# This is session-only (no DB dependency). It degrades gracefully — if
# no failures are recorded, nothing changes.
# ══════════════════════════════════════════════════════════════════════════════


def compute_rs_ranks(sym_returns: dict) -> dict:
    if not sym_returns: return {}
    syms  = list(sym_returns.keys())
    vals  = np.array([sym_returns[s] for s in syms], dtype=np.float64)
    order = np.argsort(vals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(vals))
    normalized = np.round(ranks/max(len(vals)-1,1)*100).astype(int)
    return dict(zip(syms, normalized.tolist()))

def _52w_return(close_series: pd.Series) -> float:
    if len(close_series) < 10: return 0.0
    lookback = min(252, len(close_series)-1)
    c_now  = float(close_series.iloc[-1])
    c_base = float(close_series.iloc[-lookback])
    if c_base == 0: return 0.0
    return round((c_now-c_base)/c_base*100, 2)

# ══════════════════════════════════════════════════════════════════════════════
# GAP-1 — RELATIVE LEADERSHIP INTELLIGENCE
# Analyses the RS Line (stock/index ratio) as a price series in its own right.
# Captures RS-line trend, new-highs, acceleration, and sector-relative rank.
# ══════════════════════════════════════════════════════════════════════════════

def compute_rs_leadership(
    close: pd.Series,
    nifty_close: pd.Series,
    rs_rank: int = 50,
    sector_avg_score: float = 50.0,
    stock_score: float = 50.0,
) -> dict:
    """
    Relative Leadership Intelligence (0–100 RSLeaderScore).

    Components:
    1. RS Line Slope         (25 pts) — RS line EMA slope: rising vs falling
    2. RS Line New High      (20 pts) — RS line at multi-week high vs prior peak
    3. RS Acceleration       (20 pts) — recent RS outperformance accelerating
    4. Sector-Relative Lead  (20 pts) — stock score vs sector average score
    5. RS Rank Tier          (15 pts) — percentile rank bonus

    Labels: LEADER ≥75 · IMPROVING ≥55 · NEUTRAL ≥35 · LAGGARD <35
    """
    out = dict(
        RSLeaderScore=0, RSLeaderLabel="NEUTRAL",
        RSLineSlope=0.0, RSLineHigh=False,
        RSLineSlopeRaw=0.0, RSLeaderSector=0.0,
    )
    try:
        n_stock = len(close)
        n_nifty = len(nifty_close)
        if n_stock < 30 or n_nifty < 30:
            return out

        # Align lengths
        min_len = min(n_stock, n_nifty)
        cl  = close.values[-min_len:].astype(np.float64)
        nif = nifty_close.values[-min_len:].astype(np.float64)
        n   = len(cl)

        # RS Line = stock / nifty (ratio series)
        rs_line = cl / (nif + 1e-10)

        # ── 1. RS Line Slope (0–25 pts) ─────────────────────────────────────
        slope_pts = 0.0
        rs_slope_raw = 0.0
        try:
            lb = min(20, n - 1)
            rs_ema10 = _ema_np(rs_line, 10)
            rs_ema20 = _ema_np(rs_line, 20)
            # Slope = % change in RS EMA10 over lb bars
            rs_slope_raw = (rs_ema10[-1] - rs_ema10[-lb]) / (rs_ema10[-lb] + 1e-10) * 100
            # EMA alignment on RS line itself
            rs_ema_bull = rs_ema10[-1] > rs_ema20[-1]
            if   rs_slope_raw > 5  and rs_ema_bull: slope_pts = 25.0
            elif rs_slope_raw > 2  and rs_ema_bull: slope_pts = 20.0
            elif rs_slope_raw > 0  and rs_ema_bull: slope_pts = 14.0
            elif rs_slope_raw > 0:                  slope_pts = 8.0
            elif rs_slope_raw > -2:                 slope_pts = 3.0
        except Exception:
            pass

        # ── 2. RS Line New High (0–20 pts) ──────────────────────────────────
        rslh_pts = 0.0
        rs_line_high = False
        try:
            # Compare current RS line vs 10-week (50-bar) and 6-week (30-bar) highs
            lb_rslh = min(50, n - 1)
            rs_prior_high_50 = float(np.max(rs_line[-lb_rslh:-1]))
            rs_prior_high_30 = float(np.max(rs_line[-min(30, n-1):-1]))
            rs_now = float(rs_line[-1])
            if rs_now > rs_prior_high_50:
                rslh_pts = 20.0; rs_line_high = True
            elif rs_now > rs_prior_high_30:
                rslh_pts = 13.0; rs_line_high = True
            elif rs_now > rs_prior_high_50 * 0.97:
                rslh_pts = 7.0   # within 3% of 50-bar RS high
        except Exception:
            pass

        # ── 3. RS Acceleration (0–20 pts) ───────────────────────────────────
        rsaccel_pts = 0.0
        try:
            def _rs_delta(bars):
                if n < bars + 1: return 0.0
                s = (cl[-1] - cl[-bars]) / (cl[-bars] + 1e-10) * 100
                m = (nif[-1] - nif[-bars]) / (nif[-bars] + 1e-10) * 100
                return s - m
            rs5, rs10, rs20 = _rs_delta(5), _rs_delta(10), _rs_delta(20)
            if rs5 > rs10 > rs20 > 0:        rsaccel_pts = 20.0   # triple acceleration
            elif rs5 > rs10 > 0:             rsaccel_pts = 15.0   # double acceleration
            elif rs5 > 0 and rs5 > rs20:     rsaccel_pts = 9.0    # recent vs long
            elif rs5 > 0:                    rsaccel_pts = 4.0
        except Exception:
            pass

        # ── 4. Sector-Relative Leadership (0–20 pts) ────────────────────────
        sec_pts = 0.0
        try:
            diff = stock_score - sector_avg_score
            if   diff >= 20: sec_pts = 20.0
            elif diff >= 10: sec_pts = 15.0
            elif diff >= 5:  sec_pts = 10.0
            elif diff >= 0:  sec_pts = 5.0
            elif diff >= -5: sec_pts = 2.0
        except Exception:
            pass

        # ── 5. RS Rank Tier (0–15 pts) ───────────────────────────────────────
        rank_pts = 0.0
        if   rs_rank >= 90: rank_pts = 15.0
        elif rs_rank >= 80: rank_pts = 12.0
        elif rs_rank >= 70: rank_pts = 8.0
        elif rs_rank >= 60: rank_pts = 5.0
        elif rs_rank >= 40: rank_pts = 2.0

        total = round(float(np.clip(
            slope_pts + rslh_pts + rsaccel_pts + sec_pts + rank_pts,
            0, 100
        )), 1)
        label = (
            "LEADER"   if total >= 75 else
            "IMPROVING" if total >= 55 else
            "NEUTRAL"   if total >= 35 else
            "LAGGARD"
        )
        out.update(
            RSLeaderScore   = total,
            RSLeaderLabel   = label,
            RSLineSlope     = round(slope_pts, 1),
            RSLineHigh      = rs_line_high,
            RSLineSlopeRaw  = round(rs_slope_raw, 2),
            RSLeaderSector  = round(sec_pts, 1),
        )
    except Exception:
        pass
    return out

# ══════════════════════════════════════════════════════════════════════════════
# PHASE TRANSITION MEMORY (unchanged)
# ══════════════════════════════════════════════════════════════════════════════


def compute_breadth(results):
    if not results: return {}
    total          = len(results)
    above_ema50    = sum(1 for r in results if r.get("AboveEMA50", False))
    breakout_count = sum(1 for r in results if r.get("Phase") == PHASE_BRK)
    advancing      = sum(1 for r in results if r.get("%Change", 0) > 0)
    declining      = sum(1 for r in results if r.get("%Change", 0) < 0)
    unchanged      = total-advancing-declining
    pct_above_ema50= round(above_ema50/total*100, 1)
    pct_breakout   = round(breakout_count/total*100, 1)
    ad_ratio       = round(advancing/max(declining,1), 2)
    pct_advancing  = round(advancing/total*100, 1)

    # ── Per-sector aggregation (enriched) ──────────────────────────────────
    # Groups stocks by sector, computes avg ReadinessScore (not raw Score),
    # StructureScore, TimingScore, advancing ratio, and top 3 leaders.
    from collections import defaultdict
    sec_buckets: dict = defaultdict(list)
    for r in results:
        sec = r.get("Sector", "Other")
        sec_buckets[sec].append(r)

    sector_detail = {}
    for sec, stocks in sec_buckets.items():
        n = len(stocks)
        adv = sum(1 for r in stocks if r.get("%Change", 0) > 0)
        decl = sum(1 for r in stocks if r.get("%Change", 0) < 0)
        avg_ready     = round(sum(r.get("ReadinessScore", r.get("Score", 0)) for r in stocks) / n, 1)
        avg_structure = round(sum(r.get("StructureScore", avg_ready)          for r in stocks) / n, 1)
        avg_timing    = round(sum(r.get("TimingScore",    avg_ready)          for r in stocks) / n, 1)
        avg_score     = round(sum(r.get("Score", 0)                           for r in stocks) / n, 1)
        brk_cnt       = sum(1 for r in stocks if r.get("Phase") == PHASE_BRK)
        acc_cnt       = sum(1 for r in stocks if r.get("Action") in ("BUY","STRONG BUY"))
        # Top 3 leaders: highest ReadinessScore with valid action
        leaders = sorted(
            [r for r in stocks if r.get("Action") in ("BUY","STRONG BUY","PRE-CONFIRM")],
            key=lambda x: x.get("ReadinessScore", x.get("Score", 0)), reverse=True
        )[:3]
        sector_detail[sec] = {
            "count":         n,
            "avg_ready":     avg_ready,
            "avg_structure": avg_structure,
            "avg_timing":    avg_timing,
            "avg_score":     avg_score,
            "adv_ratio":     round(adv / max(decl, 1), 2),
            "pct_adv":       round(adv / n * 100, 1),
            "breakouts":     brk_cnt,
            "actionable":    acc_cnt,
            "leaders":       [{"sym": r["Symbol"], "ready": r.get("ReadinessScore", r.get("Score",0)),
                                "action": r.get("Action",""), "ltp": r.get("LTP",0),
                                "chg": r.get("%Change", 0)} for r in leaders],
        }

    # Legacy sector_avg (Score-based) kept for backward compat
    sector_avg = {sec: d["avg_score"] for sec, d in sector_detail.items()}

    liquid_count = sum(1 for r in results if r.get("LiquidityOK", True))
    return {
        "total":total,"above_ema50":above_ema50,"pct_above_ema50":pct_above_ema50,
        "breakout_count":breakout_count,"pct_breakout":pct_breakout,
        "advancing":advancing,"declining":declining,"unchanged":unchanged,
        "ad_ratio":ad_ratio,"pct_advancing":pct_advancing,
        "sector_avg":sector_avg,"sector_detail":sector_detail,
        "liquid_count":liquid_count,
        "breadth_signal":_breadth_signal(pct_above_ema50,ad_ratio,pct_breakout),
    }

def _breadth_signal(pct_ema50, ad_ratio, pct_brk):
    score = 0
    if pct_ema50 >= 70: score += 2
    elif pct_ema50 >= 50: score += 1
    if ad_ratio >= 2.0: score += 2
    elif ad_ratio >= 1.2: score += 1
    if pct_brk >= 5: score += 1
    if score >= 4: return "STRONG","#2ecc71"
    if score >= 2: return "NEUTRAL","#f39c12"
    return "WEAK","#e74c3c"

# ══════════════════════════════════════════════════════════════════════════════
# DB / SUPABASE (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

