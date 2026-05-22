"""
risk.py — Risk-management primitives: stops, targets, position sizing, exhaustion.
"""
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

from config import (
    MODE_CFG, VIX_CALM, VIX_CAUTION, VIX_STRESS,
    LIQUIDITY_MIN_CR, NSE_SESSION_MINUTES,
    PHASE_IDLE, PHASE_SETUP, PHASE_ENTRY, PHASE_CONT, PHASE_BRK, PHASE_EXIT,
)
from indicators import atr_series

def signal_is_stale(logged_at_iso: str, mode: str) -> bool:
    try:
        validity_h = MODE_CFG[mode].get("validity_hours", 72)
        logged_at  = datetime.fromisoformat(logged_at_iso)
        return (datetime.now()-logged_at) > timedelta(hours=validity_h)
    except Exception:
        return False

def signal_age_label(logged_at_iso: str, mode: str) -> str:
    try:
        validity_h = MODE_CFG[mode].get("validity_hours", 72)
        logged_at  = datetime.fromisoformat(logged_at_iso)
        delta      = datetime.now()-logged_at
        hours      = delta.total_seconds()/3600
        stale      = hours > validity_h
        if hours < 1:   age_str = f"{int(delta.total_seconds()/60)}m ago"
        elif hours < 24: age_str = f"{hours:.1f}h ago"
        else:            age_str = f"{hours/24:.1f}d ago"
        return age_str, stale
    except Exception:
        return "unknown", False

# ══════════════════════════════════════════════════════════════════════════════
# VIX
# ══════════════════════════════════════════════════════════════════════════════


def vix_target_mult(vix_val):
    if vix_val is None or vix_val < VIX_CAUTION: return 1.0, 2.0, 3.0, 1.0
    if vix_val < VIX_STRESS:                      return 0.75, 1.4, 2.0, 1.2
    return 0.6, 1.1, 1.6, 1.35

# ══════════════════════════════════════════════════════════════════════════════
# EXPLICIT STOP-LOSS LOGIC
# Each rule is named and has a documented rationale.
# SL is NOT derived from entry logic — it is computed independently.
# ══════════════════════════════════════════════════════════════════════════════

def compute_stop_loss(
    entry: float,
    atr_val: float,
    setup_type: str,
    mode: str,
    fib: dict = None,
    sw_lo: float = None,
    support: float = None,
) -> float:
    """
    Explicit, named SL rules — one rule per setup type:

    "fib"       — SL = swing low OR Fib 61.8% minus half-ATR buffer.
                  Rationale: a fib bounce that re-enters the 61.8% has failed.

    "breakout"  — SL = entry minus 2.0×ATR (swing 2.5×, intraday 1.5×).
                  Rationale: valid breakouts should not retrace more than
                  2 ATRs; deeper means the level was false.

    "squeeze"   — SL = entry minus 1.5×ATR.
                  Rationale: squeeze releases quickly; tight stop acceptable
                  because re-entry is easy if the release fails.

    "support"   — SL = identified support level minus 0.5×ATR buffer.
                  Rationale: a clean break of support invalidates the trade.

    "default"   — SL = entry minus ATR × mode_mult, clamped between
                  closest (0.5×ATR) and furthest (atr_wide×ATR) allowed.
                  Rationale: volatility-normalised stop adapts to regime.
    """
    cfg     = MODE_CFG[mode]
    mult    = cfg["atr_mult"]
    wide    = cfg["atr_wide"]
    closest = cfg["atr_max"]

    if setup_type == "fib" and fib:
        # SL: below the 61.8% retracement with a half-ATR buffer
        base = max(float(sw_lo or 0), float(fib.get("618", entry - atr_val)))
        sl   = base - atr_val * 0.5
        # Never tighter than 0.8 ATR from entry
        sl   = min(sl, entry - atr_val * 0.8)

    elif setup_type == "breakout":
        # SL: below the breakout level by 2 ATRs (less for intraday speed)
        atr_factor = 1.5 if mode == "Intraday" else (2.5 if mode == "Positional" else 2.0)
        sl = entry - atr_val * atr_factor

    elif setup_type == "squeeze":
        # SL: tight — squeeze trades have a defined release or failure
        sl = entry - atr_val * 1.5

    elif setup_type == "support" and support is not None:
        # SL: just below identified support
        sl = support - atr_val * 0.5

    else:
        # Default: volatility-proportional, clamped
        raw_sl      = entry - atr_val * mult
        furthest_sl = entry - atr_val * wide
        closest_sl  = entry - atr_val * closest
        sl = max(furthest_sl, min(raw_sl, closest_sl))

    # Hard floor: risk can never be less than half an ATR
    min_risk = atr_val * 0.5
    if entry - sl < min_risk:
        sl = entry - min_risk

    return round(sl, 2)


# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC TARGET LOGIC
# Targets scale with momentum state (EM), exhaustion count (ExtN),
# and intrabar order-flow quality (MicroScore).
# This replaces the flat VIX-only multiplier with context-aware targets.
# ══════════════════════════════════════════════════════════════════════════════

def compute_dynamic_targets(
    entry: float,
    sl: float,
    atr_val: float,
    setup_type: str,
    fib: dict = None,
    sw_hi: float = None,
    sw_lo: float = None,
    em_label: str = "QUIET",
    ext_n: int = 0,
    micro_score: float = 50.0,
    vix_val: float = None,
    regime_bearish: bool = False,
) -> tuple:
    """
    Dynamic T1 / T2 / T3 targets.

    Base risk unit (R) = entry - SL, minimum 0.5 × ATR.

    EM multiplier table (momentum energy in the coil):
        IGNITING  → 1.5 / 3.0 / 5.0  (energy released, room to run)
        BUILDING  → 1.2 / 2.5 / 4.0  (building momentum)
        COILING   → 1.0 / 2.0 / 3.0  (baseline)
        LATENT    → 0.9 / 1.8 / 2.5  (uncertain release)
        QUIET     → 0.8 / 1.5 / 2.2  (conservative, no coil)

    ExtN compression (exhaustion reduces reward expectation):
        ext_n == 1 → compress T2/T3 by 15%
        ext_n == 2 → compress T2/T3 by 30%
        ext_n >= 3 → compress all by 40% (avoid entry anyway)

    MicroScore boost (strong intrabar flow = price has momentum):
        micro_score >= 75 → +10% on all targets
        micro_score >= 55 → +5%
        micro_score <  35 → -10%

    VIX adjustment on top (market regime context):
        applied to T1/T2/T3 as before.

    Fib/Breakout setups: Fib extension levels anchor T1/T2, dynamic
    multiplier anchors T3.
    """
    rk = max(entry - sl, atr_val * 0.5)

    # ── EM multiplier ──────────────────────────────────────────────────────
    _em_mults = {
        "IGNITING": (1.5, 3.0, 5.0),
        "BUILDING": (1.2, 2.5, 4.0),
        "COILING":  (1.0, 2.0, 3.0),
        "LATENT":   (0.9, 1.8, 2.5),
        "QUIET":    (0.8, 1.5, 2.2),
    }
    m1, m2, m3 = _em_mults.get(em_label, (1.0, 2.0, 3.0))

    # ── ExtN compression ───────────────────────────────────────────────────
    if ext_n >= 3:
        m1 *= 0.60; m2 *= 0.60; m3 *= 0.60
    elif ext_n == 2:
        m2 *= 0.70; m3 *= 0.70
    elif ext_n == 1:
        m2 *= 0.85; m3 *= 0.85

    # ── MicroScore boost ───────────────────────────────────────────────────
    if micro_score >= 75:
        m1 *= 1.10; m2 *= 1.10; m3 *= 1.10
    elif micro_score >= 55:
        m1 *= 1.05; m2 *= 1.05; m3 *= 1.05
    elif micro_score < 35:
        m1 *= 0.90; m2 *= 0.90; m3 *= 0.90

    # ── VIX adjustment ─────────────────────────────────────────────────────
    _v1, _v2, _v3, _ = vix_target_mult(vix_val)
    # Blend: dynamic mults are primary, VIX scales them
    m1 *= _v1; m2 *= _v2 / 2.0; m3 *= _v3 / 3.0  # normalise VIX to avoid double-scaling

    if regime_bearish:
        m1 *= 0.80; m2 *= 0.70; m3 *= 0.60

    # ── Fib / breakout anchoring ────────────────────────────────────────────
    if setup_type == "fib" and fib:
        t1 = round(fib.get("ext127", entry + rk * m1), 2)
        t2 = round(fib.get("ext161", entry + rk * m2), 2)
        t3 = round(fib.get("ext261", entry + rk * m3), 2)
    elif setup_type == "breakout" and fib:
        t1 = round((entry + rk * m1 + fib.get("ext127", entry + rk * m1)) / 2, 2)
        t2 = round((entry + rk * m2 + fib.get("ext161", entry + rk * m2)) / 2, 2)
        t3 = round(entry + rk * m3, 2)
    else:
        t1 = round(entry + rk * m1, 2)
        t2 = round(entry + rk * m2, 2)
        t3 = round(entry + rk * m3, 2)

    # ── Minimum move guard: T1 must be at least 0.8 ATR above entry ────────
    min_move = atr_val * 0.8
    if t1 - entry < min_move:
        t1 = round(entry + min_move, 2)
        t2 = round(entry + min_move * 2, 2)
        t3 = round(entry + min_move * 3, 2)

    return t1, t2, t3

# ══════════════════════════════════════════════════════════════════════════════
# LIQUIDITY FILTER
# ══════════════════════════════════════════════════════════════════════════════

def liquidity_ok(df, min_cr=LIQUIDITY_MIN_CR, mode="Swing"):
    try:
        traded   = df["Close"] * df["Volume"]
        n_rows   = len(df)
        if n_rows >= 2:
            try:
                delta_min = (df.index[1]-df.index[0]).total_seconds()/60
            except Exception:
                delta_min = 1440
        else:
            delta_min = 1440
        if delta_min <= 5:     bars_per_day = 75
        elif delta_min <= 15:  bars_per_day = 25
        elif delta_min <= 30:  bars_per_day = 13
        elif delta_min < 240:  bars_per_day = 7
        else:                  bars_per_day = 1
        if mode == "Intraday" and bars_per_day > 1:
            avg_daily_vol = _intraday_vol_avg(df["Volume"], bars_per_day)
            avg_cr        = float(avg_daily_vol*float(df["Close"].iloc[-1]))/1e7
        else:
            daily_traded = traded.rolling(bars_per_day).sum()
            avg_cr       = float(daily_traded.rolling(20).mean().iloc[-1])/1e7
        return avg_cr >= min_cr, round(avg_cr, 1)
    except Exception:
        return True, 0.0

# ══════════════════════════════════════════════════════════════════════════════
# SPEED-6: HTF — single shared cache, parallel pre-fetch
# ══════════════════════════════════════════════════════════════════════════════


def position_size(account_size, entry, sl, atr_val, atr_mean, vix_val,
                  risk_pct=0.02, max_capital_pct=0.20):
    risk_per_share = max(entry-sl, 0.01)
    base_qty       = int((account_size*risk_pct)/risk_per_share)
    vix_adj        = float(np.clip(20.0/vix_val, 0.5, 1.5)) if vix_val and vix_val > 0 else 1.0
    atr_adj        = float(np.clip(atr_mean/atr_val, 0.6, 1.4)) if atr_mean > 0 else 1.0
    vol_adj_qty    = max(1, int(base_qty*vix_adj*atr_adj))
    max_qty_by_cap = max(1, int((account_size*max_capital_pct)/entry))
    final_qty      = min(vol_adj_qty, max_qty_by_cap)
    return {
        "base_qty":base_qty,"vix_adj":round(vix_adj,2),"atr_adj":round(atr_adj,2),
        "vol_adj_qty":vol_adj_qty,"final_qty":final_qty,
        "capital_used":round(final_qty*entry,2),"max_loss":round(final_qty*risk_per_share,2),
        "risk_pct":risk_pct,"max_capital_pct":max_capital_pct,
        "capital_capped":final_qty < vol_adj_qty,
    }

# ══════════════════════════════════════════════════════════════════════════════
# EXHAUSTION DETECTION (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

EXT_CFG = {
    "Intraday":   dict(rsi_ceil=80,ema_dist=3.5,atr_exp=2.5,parab=3.0,clim_vol=3.0,div_bars=10),
    "Swing":      dict(rsi_ceil=78,ema_dist=3.0,atr_exp=2.5,parab=3.0,clim_vol=3.0,div_bars=14),
    "Positional": dict(rsi_ceil=75,ema_dist=2.5,atr_exp=2.0,parab=2.5,clim_vol=2.5,div_bars=20),
}
EXT_PENALTIES = {
    "rsi_overheat":-8,"atr_extension":-8,"parabolic":-6,
    "ema_distance":-5,"climactic_volume":-6,"mom_exhaustion":-4,"bearish_div":-6,
}

def detect_exhaustion(close, high, low, volume, rsi_series, e_fast_s, atr_s, atr_mean,
                      c, v, vol_avg, mode, vix_val=None):
    cfg    = EXT_CFG[mode]
    n      = len(close)
    flags  = {k:False for k in EXT_PENALTIES}
    labels = []
    rsi_ceil = cfg["rsi_ceil"]
    if vix_val is not None:
        if vix_val < VIX_CALM:     rsi_ceil += 2
        elif vix_val > VIX_STRESS: rsi_ceil -= 3
    rsi_now = float(rsi_series.iloc[-1])
    if rsi_now > rsi_ceil:
        flags["rsi_overheat"] = True; labels.append("Too hot")
    atr_val = float(atr_s.iloc[-1])
    if atr_mean > 0 and atr_val > atr_mean*cfg["atr_exp"]:
        flags["atr_extension"] = True; labels.append("Range blowout")
    if n >= 23:
        daily_pct  = close.pct_change().dropna()
        hist_sigma = float(daily_pct.iloc[-20:].std())
        exp_3b     = hist_sigma*(3**0.5)
        act_3b     = abs(float(close.iloc[-1])-float(close.iloc[-4]))/float(close.iloc[-4])
        if exp_3b > 0 and act_3b > cfg["parab"]*exp_3b:
            flags["parabolic"] = True; labels.append("Parabolic")
    e_fast_now = float(e_fast_s.iloc[-1])
    if atr_val > 0:
        ema_dist_atrs = (c-e_fast_now)/atr_val
        if ema_dist_atrs > cfg["ema_dist"]:
            flags["ema_distance"] = True; labels.append("EMA overext")
    wick_thresh = 0.35 if (c > 0 and atr_val/c > 0.03) else 0.30
    if n >= 12 and vol_avg > 0:
        prior_run = c > float(close.iloc[-11])
        up_bar    = c > float(close.iloc[-2])
        if prior_run and up_bar and v > vol_avg*cfg["clim_vol"]:
            bar_range  = float(high.iloc[-1])-float(low.iloc[-1])
            upper_wick = float(high.iloc[-1])-c
            if bar_range > 0 and (upper_wick/bar_range) > wick_thresh:
                flags["climactic_volume"] = True; labels.append("Vol climax")
    if n >= 10:
        lookback     = min(cfg["div_bars"], n-1)
        rsi_win      = rsi_series.iloc[-lookback:]
        rsi_peak     = float(rsi_win.max())
        rsi_peak_idx = rsi_win.idxmax()
        price_at_pk  = float(close[rsi_peak_idx])
        gap_req = 5 if mode == "Intraday" else 3
        if (rsi_now < rsi_peak-gap_req
                and c > price_at_pk
                and rsi_win.idxmax() != rsi_win.index[-1]):
            flags["mom_exhaustion"] = True; labels.append("Mom fade")
    if n >= 20:
        lookback  = min(cfg["div_bars"]*2, n-2)
        h_slice   = high.iloc[-lookback:]
        r_slice   = rsi_series.iloc[-lookback:]
        pivot_idx = []
        for i in range(1, len(h_slice)-1):
            if float(h_slice.iloc[i]) > float(h_slice.iloc[i-1]) and float(h_slice.iloc[i]) > float(h_slice.iloc[i+1]):
                pivot_idx.append(i)
        if len(pivot_idx) >= 2:
            p1, p2   = pivot_idx[-2], pivot_idx[-1]
            ph1, ph2 = float(h_slice.iloc[p1]), float(h_slice.iloc[p2])
            rh1, rh2 = float(r_slice.iloc[p1]), float(r_slice.iloc[p2])
            if ph2 > ph1 and rh2 < rh1-2 and (len(h_slice)-1-p2) <= 5:
                flags["bearish_div"] = True; labels.append("Bear div")
    penalty = sum(EXT_PENALTIES[k] for k,v2 in flags.items() if v2)
    n_flags = sum(flags.values())
    return flags, float(penalty), labels, n_flags

def ext_phase_override(phase, ext_flags, n_flags, mode):
    rsi_ext  = ext_flags.get("rsi_overheat", False)
    atr_ext  = ext_flags.get("atr_extension", False)
    is_crit  = n_flags >= 3 or (rsi_ext and atr_ext)
    is_mod   = n_flags == 2
    if is_crit:
        if phase == PHASE_BRK:  return PHASE_EXIT, "ext-critical→EXIT"
        if phase == PHASE_CONT: return PHASE_SETUP,"ext-critical→SETUP"
        if phase == PHASE_ENTRY:return PHASE_SETUP,"ext-critical→SETUP"
    elif is_mod:
        if phase == PHASE_BRK:  return PHASE_SETUP,"ext-moderate→SETUP"
    return phase, None

def ext_action_cap(action, n_flags, vix_val=None):
    if n_flags == 0 and (vix_val is None or vix_val < VIX_STRESS): return action
    if vix_val is not None and vix_val >= VIX_STRESS:
        return "WATCH" if action in ("STRONG BUY","BUY") else action
    if n_flags >= 3:
        return "WATCH" if action in ("STRONG BUY","BUY") else action
    return "BUY" if action == "STRONG BUY" else action

# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE MODEL (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def compute_confidence(*, norm_bull, phase, trend_up, trend_strong, vol_confirmed,
                       ema_stack, htf_aligned, regime_bullish, ext_n, vix_val,
                       phase_bonus=0, rs_rank=50):
    c  = 0.0
    c += {PHASE_BRK:20,PHASE_CONT:17,PHASE_ENTRY:13,
          PHASE_SETUP:7,PHASE_IDLE:2,PHASE_EXIT:0}.get(phase, 0)
    c += min(20, norm_bull*0.20)
    c += 15 if vol_confirmed else 5
    c += 15 if ema_stack else (7 if trend_strong else 0)
    c += 15 if htf_aligned else 0
    c += 10 if regime_bullish else 2
    c -= min(5, ext_n*2)
    if vix_val is not None and vix_val > VIX_CAUTION: c -= 5
    if rs_rank >= 90:   c += 5
    elif rs_rank >= 80: c += 3
    elif rs_rank <= 20: c -= 3
    c += phase_bonus
    return round(min(100, max(0,c)), 1)

def confidence_label(conf):
    if conf >= 80: return "HIGH","#2ecc71"
    if conf >= 60: return "MED", "#f39c12"
    if conf >= 40: return "LOW", "#e67e22"
    return "WEAK","#e74c3c"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE + ENTRY (unchanged from v14)
# ══════════════════════════════════════════════════════════════════════════════

