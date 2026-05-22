"""
scoring.py — Full signal scoring pipeline.
"""
import warnings
import logging
import time
import hashlib
import threading
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from config import (
    MODE_CFG, BULL_MAX, ACTION_THRESHOLDS, VIX_CALM, VIX_CAUTION, VIX_STRESS,
    PHASE_IDLE, PHASE_SETUP, PHASE_ENTRY, PHASE_CONT, PHASE_BRK, PHASE_EXIT,
    PHASE_COLORS, PHASE_ORDER, LIQUIDITY_MIN_CR,
    REGIME_TREND, REGIME_ROTATION, REGIME_DISTRIBUTION,
    REGIME_PANIC, REGIME_EXPANSION, _REGIME_ADJUSTMENTS,
)
from indicators import (
    _ema_np, _count_squeeze_bars, ema, rsi, atr_series, fib_levels,
    action_label, action_label_with_preconfirm, compute_emerging_score,
)
from data_fetch import fetch_async, _fetch_htf_cached, fetch_nifty
from market import compute_orthogonality_penalty
from risk import (
    compute_confidence, compute_stop_loss, compute_dynamic_targets,
    vix_target_mult, ext_phase_override, ext_action_cap, detect_exhaustion,
    signal_is_stale,
)

_phase_lock = threading.Lock()

def record_setup_failure(sym: str, setup_type: str, sector: str):
    """Call this when a position is closed at a loss to record the failure."""
    key = "setup_failure_registry"
    reg = st.session_state.get(key, [])
    reg.append({
        "sym":        sym,
        "setup_type": setup_type,
        "sector":     sector,
        "ts":         datetime.now().isoformat(),
    })
    # Keep last 50 failures only
    st.session_state[key] = reg[-50:]


def get_failure_confidence_penalty(
    sym: str,
    setup_type: str,
    sector: str,
    lookback_hours: float = 48.0,
) -> float:
    """
    Returns a confidence penalty (0–15) based on recent failures.

    Rules:
      Same symbol failed in last 48h:    -15 (strong recent memory)
      Same setup_type in same sector:    -8  (sector + setup pattern failing)
      Same setup_type generally:         -4  (setup type having a bad run)
      No recent failures:                 0
    """
    reg = st.session_state.get("setup_failure_registry", [])
    if not reg:
        return 0.0

    cutoff = datetime.now().timestamp() - lookback_hours * 3600
    recent = [
        f for f in reg
        if datetime.fromisoformat(f["ts"]).timestamp() > cutoff
    ]
    if not recent:
        return 0.0

    # Check symbol-level
    sym_failures = [f for f in recent if f["sym"] == sym]
    if sym_failures:
        return 15.0

    # Check sector + setup
    sec_setup_failures = [
        f for f in recent
        if f["sector"] == sector and f["setup_type"] == setup_type
    ]
    if len(sec_setup_failures) >= 2:
        return 8.0

    # Check setup type generally
    setup_failures = [f for f in recent if f["setup_type"] == setup_type]
    if len(setup_failures) >= 3:
        return 4.0

    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# READINESS SCORE — 3-AXIS MODEL (v18.4 final)
# Axis 1: CAUSE     — accumulation quality (PCA + SmartMoney + AccumStage)
# Axis 2: TIMING    — compression + momentum (EmScore + Micro + Squeeze + MTF)
# Axis 3: CONTEXT   — regime + sector leadership (master regime + RS leadership)
#
# ReadinessScore = Cause×0.45 + Timing×0.35 + Context×0.20
#   Cause is dominant: structural evidence is harder to fake than timing signals.
#   Context is explicit: the same setup is worth less in DISTRIBUTION than EXPANSION.
#
# Score bands (Image 4 suggestion — false precision removed):
#   80–100 → ELITE       (act with full size)
#   65–79  → STRONG      (act with normal size)
#   50–64  → DEVELOPING  (partial position / alert only)
#   < 50   → WEAK        (watch only, no entry)
# ══════════════════════════════════════════════════════════════════════════════

READY_BAND_ELITE      = 80
READY_BAND_STRONG     = 65
READY_BAND_DEVELOPING = 50


def readiness_band(score: float) -> tuple:
    """Returns (band_label, band_color, action_note)."""
    if score >= READY_BAND_ELITE:
        return "ELITE",      "#f59e0b", "Full size entry"
    if score >= READY_BAND_STRONG:
        return "STRONG",     "#22c55e", "Normal entry"
    if score >= READY_BAND_DEVELOPING:
        return "DEVELOPING", "#38bdf8", "Alert / half size"
    return     "WEAK",       "#475569", "Watch only"

# Weights
CAUSE_W_PCA    = 0.40   # PCA owns accumulation evidence
CAUSE_W_SM     = 0.35   # SmartMoney owns behaviour phase
CAUSE_W_ACCUM  = 0.25   # AccumStage owns Wyckoff position

TIMING_W_EM    = 0.35   # EmScore owns energy coil
TIMING_W_MICRO = 0.25   # Microstructure owns intrabar flow
TIMING_W_SQZ   = 0.20   # Squeeze bars own stored kinetic energy
TIMING_W_MTF   = 0.20   # MTF owns timeframe alignment

CONTEXT_W_RS   = 0.60   # RS leadership vs market/sector
CONTEXT_W_REG  = 0.40   # regime quality (EXPANSION=100, PANIC=0)

READY_CAUSE_WT   = 0.45
READY_TIMING_WT  = 0.35
READY_CONTEXT_WT = 0.20

# Regime → context score table
_REGIME_CONTEXT_SCORE = {
    REGIME_EXPANSION:    100.0,
    REGIME_TREND:         75.0,
    REGIME_ROTATION:      50.0,
    REGIME_DISTRIBUTION:  25.0,
    REGIME_PANIC:          0.0,
}

# Keep old weight constants for backward compat with any remaining references
READY_W_TREND = 0.35
READY_W_ACCUM = 0.25
READY_W_MOM   = 0.20
READY_W_RS    = 0.12
READY_W_MTF   = 0.08
STRUCT_W_PCA  = CAUSE_W_PCA
STRUCT_W_SM   = CAUSE_W_SM
STRUCT_W_ACCUM= CAUSE_W_ACCUM
TIMING_W_SQUEEZE = TIMING_W_SQZ


def compute_readiness_score(
    score: float,
    pca_score: float,
    em_score: float,
    rs_leader_score: float,
    mtf_score: float,
    smart_money_score: float = 50.0,
    accum_sequence_score: float = 50.0,
    micro_score: float = 50.0,
    sqz_bars: int = 0,
    regime: str = REGIME_TREND,
) -> dict:
    """
    3-axis ReadinessScore: Cause · Timing · Context.

    Also applies orthogonality penalty to prevent score stacking.
    Returns StructureScore (= CauseScore), TimingScore, ContextScore,
    ReadinessScore, ReadinessBand, and OrthoPenalty.
    """
    try:
        # ── Axis 1: CAUSE ────────────────────────────────────────────────────
        cause = (
            float(np.clip(pca_score,            0.0, 100.0)) * CAUSE_W_PCA   +
            float(np.clip(smart_money_score,     0.0, 100.0)) * CAUSE_W_SM    +
            float(np.clip(accum_sequence_score,  0.0, 100.0)) * CAUSE_W_ACCUM
        )

        # ── Axis 2: TIMING ───────────────────────────────────────────────────
        sqz_norm = float(np.clip(sqz_bars / 20.0 * 100.0, 0.0, 100.0))
        timing = (
            float(np.clip(em_score,    0.0, 100.0)) * TIMING_W_EM    +
            float(np.clip(micro_score, 0.0, 100.0)) * TIMING_W_MICRO +
            sqz_norm                                 * TIMING_W_SQZ   +
            float(np.clip(mtf_score,   0.0, 100.0)) * TIMING_W_MTF
        )

        # ── Axis 3: CONTEXT ──────────────────────────────────────────────────
        regime_ctx = _REGIME_CONTEXT_SCORE.get(regime, 75.0)
        rs_norm    = float(np.clip(rs_leader_score, 0.0, 100.0))
        context = (
            rs_norm    * CONTEXT_W_RS  +
            regime_ctx * CONTEXT_W_REG
        )

        # ── Orthogonality penalty ─────────────────────────────────────────────
        ortho_pen = compute_orthogonality_penalty(
            pca_score, smart_money_score, em_score, micro_score, rs_norm
        )

        raw = (
            cause   * READY_CAUSE_WT   +
            timing  * READY_TIMING_WT  +
            context * READY_CONTEXT_WT
        ) - ortho_pen

        readiness = float(np.clip(raw, 0.0, 100.0))
        band, band_col, band_note = readiness_band(readiness)

        return {
            "CauseScore":     round(float(np.clip(cause,    0.0, 100.0)), 1),
            "TimingScore":    round(float(np.clip(timing,   0.0, 100.0)), 1),
            "ContextScore":   round(float(np.clip(context,  0.0, 100.0)), 1),
            "StructureScore": round(float(np.clip(cause,    0.0, 100.0)), 1),  # alias
            "ReadinessScore": round(readiness, 1),
            "ReadinessBand":  band,
            "BandColor":      band_col,
            "BandNote":       band_note,
            "OrthoPenalty":   round(ortho_pen, 1),
        }
    except Exception:
        s = round(float(np.clip(score, 0.0, 100.0)), 1)
        b, bc, bn = readiness_band(s)
        return {
            "CauseScore": s, "TimingScore": s, "ContextScore": s,
            "StructureScore": s, "ReadinessScore": s,
            "ReadinessBand": b, "BandColor": bc, "BandNote": bn,
            "OrthoPenalty": 0.0,
        }

# ══════════════════════════════════════════════════════════════════════════════
# ENGINE RESPONSIBILITY CHARTER
# Each engine owns exactly one question. No overlap allowed.
#
#  PCA (Pre-Confirmation Accumulation)
#      OWNS: "Is institutional money building a position?"
#      Measures: CMF, hidden accumulation, effort/result, vol asymmetry,
#                failed breakdowns, range contraction sequences.
#      Does NOT measure: price momentum, timing, or trend direction.
#
#  EmScore (Emerging Momentum)
#      OWNS: "Is energy coiling and about to release?"
#      Measures: ATR compression, squeeze bars, EMA convergence,
#                RVOL acceleration, RS acceleration, opening range expansion.
#      Does NOT measure: who is buying or whether the base is valid.
#
#  SmartMoney (Behavior Model)
#      OWNS: "What phase is institutional behaviour in right now?"
#      Measures: CMF regime, block volume bias, OBV trend, absorption quality,
#                pressure asymmetry → verdict: DISTRIBUTING/NEUTRAL/ABSORBING/
#                ACCUMULATING/MARKUP_READY.
#      Does NOT measure: coil tightness or price structure.
#
#  Microstructure
#      OWNS: "What is the intrabar order-flow bias on recent bars?"
#      Measures: CLV, delta proxy, absorption ratio, wick asymmetry, VWAP dev.
#      Does NOT measure: multi-day accumulation or momentum build-up.
#
#  Institutional Volume (Engine 2)
#      OWNS: "Is today's volume pattern consistent with institutional activity?"
#      Measures: CMF, effort-vs-result, OBV trend, block-day count.
#      NOTE: CMF overlap with SmartMoney is intentional — InstEngine uses
#            it for a single-day verdict; SmartMoney uses a 20-bar regime.
#
#  Pattern Engines (Harmonic, VCP, Darvas, Fib, Candle)
#      OWNS: "Is there a high-quality price pattern completing right now?"
#      These are confirmation signals only — they never gate entry alone.
#
#  MTF Sync
#      OWNS: "Do the weekly/daily/intraday timeframes agree?"
#      Confirmation layer. Disagreement = size reduction, not skip.
# ══════════════════════════════════════════════════════════════════════════════

# ── StructureScore weights (cause layer: "why should it move?") ───────────
STRUCT_W_PCA     = 0.40   # dominant: buying-pressure fingerprint
STRUCT_W_SM      = 0.35   # smart-money behaviour phase
STRUCT_W_ACCUM   = 0.25   # Wyckoff/Weinstein stage position

# ── TimingScore weights (timing layer: "when is it ready to move?") ────────
TIMING_W_EM      = 0.35   # coil tightness + energy build
TIMING_W_MICRO   = 0.25   # intrabar order-flow (short-lag signal)
TIMING_W_SQUEEZE = 0.20   # squeeze bar count (stored kinetic energy)
TIMING_W_MTF     = 0.20   # multi-TF confirmation

# ── ReadinessScore blend (StructureScore × cause_wt + TimingScore × timing_wt)
READY_CAUSE_WT   = 0.55   # structure is the primary gate
READY_TIMING_WT  = 0.45   # timing tells you when to act




def compute_trade_intent(action: str, ext_n: int = 0, breadth_gated: bool = False) -> dict:
    """
    FIX-7 — TradeIntent: collapse all scoring into one plain-English decision.

    Tiers:
      🟢 BUY NOW   → Action in (STRONG BUY, BUY) and not severely exhausted
      🟡 NOT YET   → Action in (PRE-CONFIRM, WATCH) — building but not triggered
      🔴 IGNORE    → SKIP, or ExtN ≥ 3 (exhaustion), or breadth-gated

    Returns dict with TradeIntent, TradeIntentColor, TradeIntentIcon,
    TradeIntentDetail (one-line reason shown under the banner).
    """
    if ext_n >= 3:
        return dict(
            TradeIntent       = "IGNORE",
            TradeIntentColor  = "#ef4444",
            TradeIntentIcon   = "🔴",
            TradeIntentDetail = "Exhaustion — skip entry",
        )
    if action == "STRONG BUY":
        return dict(
            TradeIntent       = "BUY NOW",
            TradeIntentColor  = "#22c55e",
            TradeIntentIcon   = "🟢",
            TradeIntentDetail = "All conditions met — high conviction",
        )
    if action == "BUY":
        detail = "Breadth weak — size down" if breadth_gated else "Entry conditions met"
        return dict(
            TradeIntent       = "BUY NOW",
            TradeIntentColor  = "#22c55e",
            TradeIntentIcon   = "🟢",
            TradeIntentDetail = detail,
        )
    if action == "PRE-CONFIRM":
        return dict(
            TradeIntent       = "NOT YET",
            TradeIntentColor  = "#f59e0b",
            TradeIntentIcon   = "🟡",
            TradeIntentDetail = "Accumulating — wait for price trigger",
        )
    if action == "WATCH":
        return dict(
            TradeIntent       = "NOT YET",
            TradeIntentColor  = "#f59e0b",
            TradeIntentIcon   = "🟡",
            TradeIntentDetail = "Conditions building — not ready yet",
        )
    # SKIP or anything else
    return dict(
        TradeIntent       = "IGNORE",
        TradeIntentColor  = "#ef4444",
        TradeIntentIcon   = "🔴",
        TradeIntentDetail = "No setup — move on",
    )




def fmt(val):
    """Format a price/rupee value for display; returns '—' for None/NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"₹{val:,.2f}"

def _session_elapsed_fraction() -> float:
    now_ist  = datetime.utcnow() + timedelta(hours=5, minutes=30)
    minutes_since_open = (now_ist.hour*60+now_ist.minute)-(NSE_OPEN_HOUR*60+NSE_OPEN_MIN)
    fraction = minutes_since_open / NSE_SESSION_MINUTES
    return float(np.clip(fraction, 0.05, 1.0))

def _intraday_vol_avg(volume: pd.Series, bars_per_day: int) -> float:
    elapsed_frac = _session_elapsed_fraction()
    today_bars   = int(min(bars_per_day*elapsed_frac+1, len(volume)))
    today_vol    = float(volume.iloc[-today_bars:].sum())
    today_proj   = today_vol / elapsed_frac
    if len(volume) > bars_per_day + today_bars:
        prior = volume.iloc[:-(today_bars)].rolling(bars_per_day).sum().dropna()
        prior_daily = prior.iloc[-5:].values.tolist()
    else:
        prior_daily = []
    all_days = prior_daily + [today_proj]
    return float(np.mean(all_days)) if all_days else float(volume.mean()*bars_per_day)

# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL VALIDITY
# ══════════════════════════════════════════════════════════════════════════════


def record_phase_transition(sym: str, new_phase: str):
    if "phase_history" not in st.session_state:
        st.session_state["phase_history"] = {}
    history = st.session_state["phase_history"]
    if sym not in history: history[sym] = []
    prev_phase = history[sym][-1][1] if history[sym] else None
    changed = prev_phase != new_phase
    is_prog = is_regr = False
    arrow   = ""
    if changed:
        ts = datetime.now().isoformat()
        history[sym].append((ts, new_phase))
        history[sym] = history[sym][-10:]
        if prev_phase is not None:
            prev_ord = PHASE_ORDER.get(prev_phase, 0)
            new_ord  = PHASE_ORDER.get(new_phase, 0)
            if new_phase == PHASE_EXIT:             arrow="→EXIT"; is_regr=True
            elif new_ord > prev_ord:                arrow=f"↗{new_phase}"; is_prog=True
            elif new_ord < prev_ord and new_phase!=PHASE_EXIT: arrow=f"↘{new_phase}"; is_regr=True
    return changed, arrow, is_prog, is_regr

def phase_transition_conf_bonus(sym: str) -> int:
    history = st.session_state.get("phase_history", {})
    if sym not in history or len(history[sym]) < 3: return 0
    last3 = [h[1] for h in history[sym][-3:]]
    progressions = [
        [PHASE_SETUP,PHASE_ENTRY,PHASE_CONT],
        [PHASE_ENTRY,PHASE_CONT,PHASE_BRK],
        [PHASE_SETUP,PHASE_ENTRY,PHASE_BRK],
    ]
    return 5 if last3 in progressions else 0

def get_phase_arrow(sym: str) -> str:
    history = st.session_state.get("phase_history", {})
    if sym not in history or len(history[sym]) < 2: return ""
    prev = history[sym][-2][1]; curr = history[sym][-1][1]
    if curr == PHASE_EXIT:                                   return "→EXIT"
    if PHASE_ORDER.get(curr,0) > PHASE_ORDER.get(prev,0):  return "↗"
    if PHASE_ORDER.get(curr,0) < PHASE_ORDER.get(prev,0):  return "↘"
    return ""

# ══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING (FIX-5, unchanged)
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

def detect_phase_and_entry(df, mode, *, c, e_fast_s, e_slow_s, atr_s,
                            atr_val, atr_mean, v, vol_avg, fib, sw_hi, sw_lo,
                            in_golden, near_e127, near_e161, norm_bull,
                            trend_up, trend_down, trend_strong, score_th,
                            vdu_setup=False, htf_up=True,
                            regime_bearish=False, vix_val=None):
    cfg   = MODE_CFG[mode]
    close = df["Close"]; high = df["High"]; n = len(close)
    if n < 60: return PHASE_IDLE, None, "norm"
    e_fast_val = float(e_fast_s.iloc[-1])
    e_slow_val = float(e_slow_s.iloc[-1])
    brk_lb     = 5
    rolling_hi_brk = float(high.iloc[-brk_lb-1:-1].max()) if n > brk_lb+1 else float(high.iloc[-1])
    buf = atr_val*0.15
    is_compressed = atr_val < atr_mean*0.8
    is_expanding  = atr_val > float(atr_s.iloc[-2])
    prior_3bar_atr_expanded = atr_val > atr_mean*1.4
    body = (abs(float(close.iloc[-1])-float(df["Open"].iloc[-1]))
            if "Open" in df.columns else atr_val*0.3)
    upper_wick = (float(high.iloc[-1])-max(float(close.iloc[-1]),float(df["Open"].iloc[-1]))
                  if "Open" in df.columns else 0)
    is_exhaustion = upper_wick > body*1.5
    brk_vol_ok = (v > vol_avg*1.5) if vol_avg > 0 else False
    vol_spike  = v > vol_avg*1.3
    is_fib_buy = trend_up and in_golden
    cont_vol_mult = 1.5 if (regime_bearish or (vix_val and vix_val>VIX_CAUTION)) else 1.2
    BRK_CONF_MIN  = 0.70 if regime_bearish else 0.65
    brk_weights = {
        "price_above_high":(0.35, c > rolling_hi_brk+buf),
        "score_ok":        (0.20, norm_bull >= score_th),
        "compressed":      (0.20, is_compressed),
        "expanding":       (0.15, is_expanding),
        "vol_spike":       (0.10, vol_spike),
    }
    brk_confidence = sum(w for w,cond in brk_weights.values() if cond)
    is_breakout = (brk_confidence >= BRK_CONF_MIN and not is_exhaustion
                   and brk_vol_ok and not prior_3bar_atr_expanded and htf_up)
    was_recent_brk = False; recent_brk_bar = None
    if not is_breakout and n > brk_lb*2+2:
        for k in range(1, brk_lb+1):
            look_start = -(brk_lb+1+k); look_end = -(1+k)
            if abs(look_start) > n or abs(look_end) > n: break
            prev_rolling_hi = float(high.iloc[look_start:look_end].max())
            prev_hi_k       = float(high.iloc[-k])
            prev_close_k    = float(close.iloc[-k])
            prev_vol_k      = float(df["Volume"].iloc[-k])
            close_above_brk = prev_close_k > prev_rolling_hi
            prev_open_k     = (float(df["Open"].iloc[-k]) if "Open" in df.columns else prev_close_k)
            body_non_red    = prev_close_k >= prev_open_k
            hist_vol        = df["Volume"].iloc[:-k]
            hist_avg_k      = (float(hist_vol.rolling(20).mean().iloc[-1])
                               if len(hist_vol) >= 20 else vol_avg)
            vol_gate        = (hist_avg_k == 0 or prev_vol_k > hist_avg_k*1.5)
            if (prev_hi_k > prev_rolling_hi+buf and close_above_brk
                    and body_non_red and vol_gate):
                was_recent_brk = True; recent_brk_bar = k; break
    is_cont = (n >= 4 and c > float(close.iloc[-4:-1].max())
               and c > e_fast_val and v > vol_avg*cont_vol_mult
               and trend_strong and htf_up)
    ema_down    = e_fast_val < e_slow_val and float(e_fast_s.iloc[-4]) < float(e_slow_s.iloc[-4])
    trail_level = float(close.iloc[-10:].max())-atr_val*1.5
    trail_break = c < trail_level
    if trend_down and ema_down:     phase, setup_type = PHASE_EXIT, "norm"
    elif is_breakout:               phase, setup_type = PHASE_BRK, "breakout"
    elif was_recent_brk and trend_strong:
        phase, setup_type = (PHASE_CONT,"breakout") if trend_up else (PHASE_SETUP,"breakout")
    elif (is_fib_buy or norm_bull >= score_th) and is_cont and trend_up:
        phase, setup_type = PHASE_CONT, ("fib" if is_fib_buy else "norm")
    elif (is_fib_buy or norm_bull >= score_th) and trend_up:
        phase, setup_type = PHASE_ENTRY, ("fib" if is_fib_buy else "norm")
    elif (is_fib_buy or norm_bull >= score_th*0.85 or vdu_setup) and trend_up:
        phase, setup_type = PHASE_SETUP, ("fib" if is_fib_buy else ("vdu" if vdu_setup else "norm"))
    elif trail_break and trend_up:  phase, setup_type = PHASE_EXIT, "norm"
    else:                           phase, setup_type = PHASE_IDLE, "norm"
    if not htf_up and phase in (PHASE_ENTRY,PHASE_CONT,PHASE_BRK):
        phase, setup_type = PHASE_SETUP, setup_type
    entry_price = None
    if phase in (PHASE_ENTRY,PHASE_CONT,PHASE_BRK,PHASE_SETUP):
        prox = atr_val*0.3
        if is_breakout:         entry_price = round(rolling_hi_brk+buf, 2)
        elif was_recent_brk:    entry_price = round(c, 2)
        elif is_fib_buy and fib: entry_price = round(fib["618"]+prox*0.3, 2)
        else:
            cross       = close > e_fast_s
            signal_bars = cross & ~cross.shift(1).fillna(False)
            if signal_bars.any():
                last_cross_idx = signal_bars[::-1].idxmax()
                cross_pos      = close.index.get_loc(last_cross_idx)
                bars_ago       = (n-1)-cross_pos
                cross_px       = float(close[last_cross_idx])
                entry_price = round(cross_px,2) if (bars_ago<=10 and cross_px>=c*0.97) else round(c,2)
            else:
                entry_price = round(c, 2)
    return phase, entry_price, setup_type

# ══════════════════════════════════════════════════════════════════════════════
# TARGET COMPUTATION (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_targets(entry, sl, atr_val, fib, setup_type, sw_hi, sw_lo,
                     regime_bearish=False, vix_val=None):
    rk = max(entry-sl, atr_val*0.5)
    t1m,t2m,t3m,sl_exp = vix_target_mult(vix_val)
    if regime_bearish: t1m*=0.8; t2m*=0.7; t3m*=0.6
    if setup_type == "fib" and fib:
        t1=round(fib["ext127"],2); t2=round(fib["ext161"],2)
        ext_r=fib["ext161"]-fib["ext127"]
        t3=round(fib["ext161"]+min(ext_r,atr_val*3),2)
    elif setup_type == "breakout" and fib:
        t1=round((entry+rk*t1m+fib["ext127"])/2,2)
        t2=round((entry+rk*t2m+fib["ext161"])/2,2)
        t3=round((entry+rk*t3m+fib["ext261"])/2,2)
    else:
        t1=round(entry+rk*t1m,2); t2=round(entry+rk*t2m,2); t3=round(entry+rk*t3m,2)
    min_move = atr_val*0.8
    if t1-entry < min_move:
        t1=round(entry+min_move,2); t2=round(entry+min_move*2,2); t3=round(entry+min_move*3,2)
    return t1, t2, t3, sl_exp

# ══════════════════════════════════════════════════════════════════════════════
# NIFTY FETCH
# ══════════════════════════════════════════════════════════════════════════════


def compute_preconfirmation_accumulation(
    df: pd.DataFrame,
    mode: str,
) -> dict:
    """
    Pre-Confirmation Accumulation Score (0–100).

    Components (max pts):
    1. Relative CMF          15 — CMF improving vs its own rolling baseline
    2. Vol Compression Seq   15 — sequence of tightening ATR/range contractions
    3. Hidden Accumulation   15 — down-days low-vol, up-days high-vol (absorption)
    4. Effort vs Result      15 — big volume that barely moves price = supply absorbed
    5. Range Contraction     10 — persistent NR bars below historical avg range
    6. Failed Breakdown      15 — wicks below support closed back above = buying
    7. Volume Asymmetry      15 — up-day volume persistently > down-day volume

    Labels: ACCUMULATING ≥65 · BUILDING ≥50 · FORMING ≥35 · WEAK ≥20 · NONE <20
    """
    out = dict(
        PCAScore=0.0, PCALabel="NONE",
        PCACMFRel=0.0, PCAVolCmpSeq=0.0, PCAHiddenAccum=0.0,
        PCAEffortResult=0.0, PCARangeCont=0.0, PCAFailedBrkdn=0.0, PCAVolAsym=0.0,
    )
    try:
        if df is None or len(df) < 30:
            return out

        cl  = df["Close"].values.astype(np.float64)
        hi  = df["High"].values.astype(np.float64)
        lo  = df["Low"].values.astype(np.float64)
        vol = df["Volume"].values.astype(np.float64)
        n   = len(cl)
        op  = (df["Open"].values.astype(np.float64)
               if "Open" in df.columns else cl.copy())

        # ── 1. RELATIVE CMF (0–15 pts) ────────────────────────────────────────
        # CMF improving vs its own 40-bar prior baseline = fresh buying pressure
        cmf_pts = 0.0
        try:
            win = min(20, n)
            hlr = np.where((hi[-win:] - lo[-win:]) == 0, 1e-10, hi[-win:] - lo[-win:])
            mfm = ((cl[-win:] - lo[-win:]) - (hi[-win:] - cl[-win:])) / hlr
            cmf_now = float(np.sum(mfm * vol[-win:]) / (np.sum(vol[-win:]) + 1e-10))

            if n >= 40:
                hlr_old = np.where(
                    (hi[-40:-20] - lo[-40:-20]) == 0, 1e-10,
                    hi[-40:-20] - lo[-40:-20]
                )
                mfm_old = ((cl[-40:-20] - lo[-40:-20]) -
                           (hi[-40:-20] - cl[-40:-20])) / hlr_old
                cmf_old = float(
                    np.sum(mfm_old * vol[-40:-20]) / (np.sum(vol[-40:-20]) + 1e-10)
                )
                delta = cmf_now - cmf_old
                if   cmf_now > 0.15 and delta > 0:        cmf_pts = 15.0
                elif cmf_now > 0.05 and delta > 0:         cmf_pts = 11.0
                elif cmf_now > 0    and delta > 0.05:      cmf_pts = 8.0
                elif cmf_now > 0:                          cmf_pts = 4.0
                elif cmf_now > -0.05 and delta > 0.10:    cmf_pts = 3.0
            else:
                if   cmf_now > 0.15: cmf_pts = 10.0
                elif cmf_now > 0.05: cmf_pts = 6.0
                elif cmf_now > 0:    cmf_pts = 3.0
        except Exception:
            pass

        # ── 2. VOLATILITY COMPRESSION SEQUENCING (0–15 pts) ──────────────────
        # Consecutive narrowing ranges AND multi-window avg sequence tightening
        vc_pts = 0.0
        try:
            if n >= 10:
                ranges = hi - lo
                consec = 0
                for k in range(1, min(15, n - 1)):
                    if ranges[-k] < ranges[-(k + 1)]:
                        consec += 1
                    else:
                        break
                avg5  = float(np.mean(ranges[-5:]))  if n >= 5  else float(np.mean(ranges))
                avg10 = float(np.mean(ranges[-10:])) if n >= 10 else avg5
                avg20 = float(np.mean(ranges[-20:])) if n >= 20 else avg10
                sequenced = avg5 < avg10 < avg20

                if   consec >= 8 and sequenced: vc_pts = 15.0
                elif consec >= 5 and sequenced: vc_pts = 12.0
                elif consec >= 3:               vc_pts = 8.0
                elif consec >= 2 and sequenced: vc_pts = 5.0
                elif sequenced:                 vc_pts = 3.0
        except Exception:
            pass

        # ── 3. HIDDEN ACCUMULATION (0–15 pts) ────────────────────────────────
        # Up-day vol >> down-day vol even while price is flat/slightly negative
        ha_pts = 0.0
        try:
            lb = min(20, n - 1)
            up_vols   = [float(vol[i]) for i in range(-lb, 0) if cl[i] > cl[i - 1]]
            down_vols = [float(vol[i]) for i in range(-lb, 0) if cl[i] <= cl[i - 1]]
            if up_vols and down_vols:
                avg_up   = float(np.mean(up_vols))
                avg_down = float(np.mean(down_vols))
                ratio    = avg_up / (avg_down + 1e-10)
                price_chg = (float(cl[-1]) - float(cl[-lb])) / (float(cl[-lb]) + 1e-10) * 100

                if   ratio >= 2.5:                          ha_pts = 15.0
                elif ratio >= 2.0:                          ha_pts = 12.0
                elif ratio >= 1.5:                          ha_pts = 9.0
                elif ratio >= 1.2:                          ha_pts = 6.0
                elif ratio >= 1.05 and price_chg < 2.0:   ha_pts = 3.0
        except Exception:
            pass

        # ── 4. EFFORT VS RESULT ANOMALIES (0–15 pts) ─────────────────────────
        # High volume + tiny body = supply absorbed (Wyckoff spring fingerprint)
        evr_pts = 0.0
        try:
            lb_evr   = min(10, n - 1)
            avg_vol_ = float(np.mean(vol[-21:-1])) if n >= 22 else float(np.mean(vol[:-1]))
            avg_rng_ = (float(np.mean(hi[-21:-1] - lo[-21:-1]))
                        if n >= 22 else float(np.mean(hi - lo)))
            if avg_vol_ > 0 and avg_rng_ > 0:
                anomalies = 0
                for k in range(1, lb_evr + 1):
                    v_k    = float(vol[-k])
                    body_k = abs(float(cl[-k]) - float(op[-k]))
                    cl_k   = float(cl[-k])
                    op_k   = float(op[-k])
                    hi_k   = float(hi[-k])
                    lo_k   = float(lo[-k])
                    mid_k  = (hi_k + lo_k) / 2
                    # High-vol tiny-body (absorption bar)
                    if v_k > avg_vol_ * 1.3 and body_k < avg_rng_ * 0.30:
                        anomalies += 1
                    # High-vol red day that closed upper half (buyers absorbed selling)
                    elif (v_k > avg_vol_ * 1.5 and cl_k < op_k
                          and cl_k > mid_k):
                        anomalies += 1

                if   anomalies >= 5: evr_pts = 15.0
                elif anomalies >= 3: evr_pts = 11.0
                elif anomalies >= 2: evr_pts = 7.0
                elif anomalies >= 1: evr_pts = 4.0
        except Exception:
            pass

        # ── 5. RANGE CONTRACTION PERSISTENCE (0–10 pts) ──────────────────────
        # How many of last 10 bars have range below 85% of 20-bar historical avg
        rc_pts = 0.0
        try:
            if n >= 12:
                ranges_ = hi - lo
                avg_rng20 = float(np.mean(ranges_[-21:-1])) if n >= 22 else float(np.mean(ranges_[:-1]))
                if avg_rng20 > 0:
                    nr_count = int(np.sum(ranges_[-10:] < avg_rng20 * 0.85))
                    if   nr_count >= 9: rc_pts = 10.0
                    elif nr_count >= 7: rc_pts = 8.0
                    elif nr_count >= 5: rc_pts = 6.0
                    elif nr_count >= 3: rc_pts = 4.0
                    elif nr_count >= 2: rc_pts = 2.0
        except Exception:
            pass

        # ── 6. FAILED BREAKDOWN ABSORPTION (0–15 pts) ────────────────────────
        # Bars where lows pierced recent support but closed back above = buying
        fba_pts = 0.0
        try:
            if n >= 15:
                lb_fba  = min(10, n - 5)
                support = float(np.min(lo[-lb_fba - 5:-5]))   # recent swing support
                fba_cnt = 0
                for k in range(1, lb_fba + 1):
                    lo_k = float(lo[-k]); cl_k = float(cl[-k])
                    if lo_k < support * 0.997 and cl_k >= support * 0.999:
                        fba_cnt += 1

                if   fba_cnt >= 4: fba_pts = 15.0
                elif fba_cnt >= 3: fba_pts = 12.0
                elif fba_cnt >= 2: fba_pts = 8.0
                elif fba_cnt >= 1: fba_pts = 5.0

                # Bonus: if holding above support on tight range (coil above base)
                if fba_cnt >= 1 and float(cl[-1]) > support:
                    rng_now = float(hi[-1] - lo[-1])
                    avg_rng_b = float(np.mean(hi[-20:] - lo[-20:])) if n >= 20 else rng_now
                    if avg_rng_b > 0 and rng_now < avg_rng_b * 0.70:
                        fba_pts = min(15.0, fba_pts + 3.0)
        except Exception:
            pass

        # ── 7. VOLUME ASYMMETRY (0–15 pts) ────────────────────────────────────
        # Up-day vol is consistently AND significantly > down-day vol
        va_pts = 0.0
        try:
            lb_va       = min(30, n - 1)
            up_vols_a   = [float(vol[i]) for i in range(-lb_va, 0) if cl[i] > cl[i - 1]]
            down_vols_a = [float(vol[i]) for i in range(-lb_va, 0) if cl[i] < cl[i - 1]]
            if len(up_vols_a) >= 3 and len(down_vols_a) >= 3:
                median_down  = float(np.median(down_vols_a))
                pct_above    = (sum(1 for v in up_vols_a if v > median_down)
                                / len(up_vols_a))
                mag_ratio    = (float(np.mean(up_vols_a))
                                / (float(np.mean(down_vols_a)) + 1e-10))

                if   pct_above >= 0.80 and mag_ratio >= 1.8: va_pts = 15.0
                elif pct_above >= 0.70 and mag_ratio >= 1.5: va_pts = 12.0
                elif pct_above >= 0.65 and mag_ratio >= 1.3: va_pts = 9.0
                elif pct_above >= 0.55 and mag_ratio >= 1.2: va_pts = 6.0
                elif pct_above >= 0.50 and mag_ratio >= 1.1: va_pts = 3.0
        except Exception:
            pass

        # ── TOTAL ──────────────────────────────────────────────────────────────
        total = round(float(np.clip(
            cmf_pts + vc_pts + ha_pts + evr_pts + rc_pts + fba_pts + va_pts,
            0, 100
        )), 1)
        label = (
            "ACCUMULATING" if total >= 65 else
            "BUILDING"     if total >= 50 else
            "FORMING"      if total >= 35 else
            "WEAK"         if total >= 20 else
            "NONE"
        )
        out.update(
            PCAScore        = total,
            PCALabel        = label,
            PCACMFRel       = round(cmf_pts,  1),
            PCAVolCmpSeq    = round(vc_pts,   1),
            PCAHiddenAccum  = round(ha_pts,   1),
            PCAEffortResult = round(evr_pts,  1),
            PCARangeCont    = round(rc_pts,   1),
            PCAFailedBrkdn  = round(fba_pts,  1),
            PCAVolAsym      = round(va_pts,   1),
        )
    except Exception:
        pass
    return out

# ══════════════════════════════════════════════════════════════════════════════
# GAP-2 — SMART MONEY BEHAVIOR MODEL
# Synthesizes all buying-pressure signals into one behavioral verdict.
# Goes beyond PCA (snapshot score) to classify the behaviour type.
# ══════════════════════════════════════════════════════════════════════════════

_SM_VERDICTS = ["DISTRIBUTING", "NEUTRAL", "ABSORBING", "ACCUMULATING", "MARKUP_READY"]

def compute_smart_money_model(
    df: pd.DataFrame,
    mode: str,
    pca_score: float = 0.0,
    inst_score: float = 50.0,
    obv_trend: bool = True,
) -> dict:
    """
    Smart Money Behavior Model (0–100 SmartMoneyScore).

    Components:
    1. CMF Regime          (20 pts) — current CMF + trend direction
    2. Block Volume        (20 pts) — large-lot days (>2× avg vol) net bias
    3. OBV Trend           (15 pts) — OBV slope and EMA alignment
    4. Absorption Quality  (25 pts) — high-vol days that barely moved price
    5. Pressure Asymmetry  (20 pts) — net buying pressure over last N bars

    Verdicts: MARKUP_READY ≥80 · ACCUMULATING ≥65 · ABSORBING ≥50 ·
              NEUTRAL ≥35 · DISTRIBUTING <35
    """
    out = dict(
        SmartMoneyScore=0.0, SmartMoneyVerdict="NEUTRAL",
        SMBehaviorPhase="UNKNOWN", SMConfidence=0,
        SMCMFScore=0.0, SMBlockScore=0.0,
        SMOBVScore=0.0, SMAbsorptionScore=0.0, SMPressureScore=0.0,
    )
    try:
        if df is None or len(df) < 30:
            return out
        cl  = df["Close"].values.astype(np.float64)
        hi  = df["High"].values.astype(np.float64)
        lo  = df["Low"].values.astype(np.float64)
        vol = df["Volume"].values.astype(np.float64)
        op  = (df["Open"].values.astype(np.float64)
               if "Open" in df.columns else cl.copy())
        n = len(cl)

        # ── 1. CMF Regime (0–20 pts) ─────────────────────────────────────────
        cmf_pts = 0.0
        try:
            win = min(20, n)
            hlr = np.where((hi[-win:] - lo[-win:]) == 0, 1e-10, hi[-win:] - lo[-win:])
            mfm = ((cl[-win:] - lo[-win:]) - (hi[-win:] - cl[-win:])) / hlr
            cmf_now = float(np.sum(mfm * vol[-win:]) / (np.sum(vol[-win:]) + 1e-10))
            # CMF trend: compare two halves of the window
            half = win // 2
            if half > 2:
                hlr1 = np.where((hi[-win:-half]-lo[-win:-half])==0, 1e-10, hi[-win:-half]-lo[-win:-half])
                mfm1 = ((cl[-win:-half]-lo[-win:-half])-(hi[-win:-half]-cl[-win:-half]))/hlr1
                cmf_old = float(np.sum(mfm1*vol[-win:-half])/(np.sum(vol[-win:-half])+1e-10))
                cmf_rising = cmf_now > cmf_old
            else:
                cmf_rising = cmf_now > 0
            if   cmf_now > 0.20 and cmf_rising: cmf_pts = 20.0
            elif cmf_now > 0.10 and cmf_rising: cmf_pts = 16.0
            elif cmf_now > 0.05 and cmf_rising: cmf_pts = 12.0
            elif cmf_now > 0    and cmf_rising: cmf_pts = 8.0
            elif cmf_now > 0:                   cmf_pts = 5.0
            elif cmf_now > -0.05 and cmf_rising:cmf_pts = 3.0
        except Exception:
            pass

        # ── 2. Block Volume Net Bias (0–20 pts) ──────────────────────────────
        block_pts = 0.0
        try:
            avg_vol = float(np.mean(vol[-21:-1])) if n >= 22 else float(np.mean(vol))
            if avg_vol > 0:
                threshold = avg_vol * 2.0
                bull_blocks = sum(
                    1 for i in range(-min(20, n-1), 0)
                    if vol[i] >= threshold and cl[i] >= op[i]
                )
                bear_blocks = sum(
                    1 for i in range(-min(20, n-1), 0)
                    if vol[i] >= threshold and cl[i] < op[i]
                )
                net = bull_blocks - bear_blocks
                if   net >= 5: block_pts = 20.0
                elif net >= 3: block_pts = 15.0
                elif net >= 2: block_pts = 10.0
                elif net >= 1: block_pts = 6.0
                elif net == 0: block_pts = 3.0
        except Exception:
            pass

        # ── 3. OBV Trend (0–15 pts) ──────────────────────────────────────────
        obv_pts = 0.0
        try:
            obv = np.zeros(n)
            for i in range(1, n):
                if cl[i] > cl[i-1]:   obv[i] = obv[i-1] + vol[i]
                elif cl[i] < cl[i-1]: obv[i] = obv[i-1] - vol[i]
                else:                 obv[i] = obv[i-1]
            obv_ema10 = _ema_np(obv, 10)
            obv_ema20 = _ema_np(obv, 20)
            obv_slope = (obv_ema10[-1] - obv_ema10[-min(10, n-1)]) / (abs(obv_ema10[-min(10, n-1)]) + 1e-10)
            obv_bull  = obv_ema10[-1] > obv_ema20[-1]
            if   obv_slope > 0.10 and obv_bull: obv_pts = 15.0
            elif obv_slope > 0.03 and obv_bull: obv_pts = 11.0
            elif obv_slope > 0    and obv_bull: obv_pts = 7.0
            elif obv_slope > 0:                 obv_pts = 4.0
            elif obv_trend:                     obv_pts = 5.0  # fallback from inst engine
        except Exception:
            if obv_trend: obv_pts = 5.0

        # ── 4. Absorption Quality (0–25 pts) ─────────────────────────────────
        # High-vol bars where price barely moved = supply being absorbed by buyers
        abs_pts = 0.0
        try:
            avg_vol_ = float(np.mean(vol[-21:-1])) if n >= 22 else float(np.mean(vol))
            avg_rng_ = float(np.mean(hi[-21:-1] - lo[-21:-1])) if n >= 22 else float(np.mean(hi - lo))
            if avg_vol_ > 0 and avg_rng_ > 0:
                quality_abs = 0
                strong_abs  = 0
                for k in range(1, min(15, n)):
                    v_k    = float(vol[-k])
                    body_k = abs(float(cl[-k]) - float(op[-k]))
                    hi_k   = float(hi[-k]); lo_k = float(lo[-k])
                    cl_k   = float(cl[-k]); op_k = float(op[-k])
                    mid_k  = (hi_k + lo_k) / 2
                    # High-vol tiny-body = absorption
                    if v_k > avg_vol_ * 1.5 and body_k < avg_rng_ * 0.25:
                        quality_abs += 1
                        if cl_k > mid_k:  # closed upper half = buyers won
                            strong_abs += 1
                    # High-vol red close in upper half = sellers tried, buyers held
                    elif (v_k > avg_vol_ * 1.8 and cl_k < op_k
                          and cl_k > mid_k and body_k < avg_rng_ * 0.40):
                        quality_abs += 1
                if   strong_abs >= 3: abs_pts = 25.0
                elif strong_abs >= 2: abs_pts = 20.0
                elif quality_abs >= 4:abs_pts = 18.0
                elif quality_abs >= 3:abs_pts = 14.0
                elif quality_abs >= 2:abs_pts = 9.0
                elif quality_abs >= 1:abs_pts = 5.0
        except Exception:
            pass

        # ── 5. Pressure Asymmetry (0–20 pts) ─────────────────────────────────
        # Net directional pressure per bar: (close - open) / range × volume
        pres_pts = 0.0
        try:
            lb_p = min(20, n - 1)
            rng_arr = hi[-lb_p:] - lo[-lb_p:]
            # Buying pressure fraction per bar (0=full selling, 1=full buying)
            bpf = np.where(rng_arr > 0,
                           (cl[-lb_p:] - lo[-lb_p:]) / rng_arr,
                           0.5)
            # Weighted by volume
            total_vol = float(np.sum(vol[-lb_p:]))
            if total_vol > 0:
                net_bp = float(np.sum((bpf - 0.5) * vol[-lb_p:])) / total_vol  # -0.5 to +0.5
                if   net_bp > 0.20: pres_pts = 20.0
                elif net_bp > 0.12: pres_pts = 15.0
                elif net_bp > 0.06: pres_pts = 10.0
                elif net_bp > 0.02: pres_pts = 6.0
                elif net_bp > 0:    pres_pts = 3.0
        except Exception:
            pass

        # Blend with PCA and inst_score
        pca_boost = round(float(np.clip((pca_score - 35.0) / 65.0 * 10.0, 0, 10)), 1)
        inst_boost = round(float(np.clip((inst_score - 50.0) / 50.0 * 5.0, 0, 5)), 1)

        raw_total = cmf_pts + block_pts + obv_pts + abs_pts + pres_pts + pca_boost + inst_boost
        total = round(float(np.clip(raw_total, 0, 100)), 1)

        if   total >= 80: verdict = "MARKUP_READY"
        elif total >= 65: verdict = "ACCUMULATING"
        elif total >= 50: verdict = "ABSORBING"
        elif total >= 35: verdict = "NEUTRAL"
        else:             verdict = "DISTRIBUTING"

        # Behavioral phase narrative
        if verdict in ("MARKUP_READY", "ACCUMULATING"):
            phase = "INSTITUTIONAL_BUY"
        elif verdict == "ABSORBING":
            phase = "SUPPLY_ABSORPTION"
        elif verdict == "NEUTRAL":
            phase = "EQUILIBRIUM"
        else:
            phase = "SUPPLY_PRESSURE"

        confidence = min(100, int(
            (1 if cmf_pts > 8 else 0) +
            (1 if block_pts > 6 else 0) +
            (1 if obv_pts > 7 else 0) +
            (1 if abs_pts > 9 else 0) +
            (1 if pres_pts > 6 else 0)
        ) * 20)

        out.update(
            SmartMoneyScore    = total,
            SmartMoneyVerdict  = verdict,
            SMBehaviorPhase    = phase,
            SMConfidence       = confidence,
            SMCMFScore         = round(cmf_pts,   1),
            SMBlockScore       = round(block_pts,  1),
            SMOBVScore         = round(obv_pts,    1),
            SMAbsorptionScore  = round(abs_pts,    1),
            SMPressureScore    = round(pres_pts,   1),
        )
    except Exception:
        pass
    return out

# ══════════════════════════════════════════════════════════════════════════════
# GAP-3 — ACCUMULATION SEQUENCING
# Infers WHERE in the Wyckoff / Weinstein base-building sequence a stock is.
# Uses PCA + EmScore + Phase + price structure to determine sequence position.
# ══════════════════════════════════════════════════════════════════════════════

_ACCUM_STAGES = {
    "NONE":  (0,  "No base detected"),
    "1A":    (20, "Base building — early contraction"),
    "1B":    (40, "Base testing — support holding"),
    "1C":    (60, "Spring / re-test — buyers absorbing supply"),
    "2A":    (80, "Early markup — first leg out of base"),
    "2B":    (95, "Markup continuation — trend established"),
}

def compute_accumulation_sequence(
    df: pd.DataFrame,
    mode: str,
    pca_score: float = 0.0,
    em_score: float = 0.0,
    phase: str = "IDLE",
    smart_money_verdict: str = "NEUTRAL",
    rs_line_high: bool = False,
) -> dict:
    """
    Accumulation Sequence Detector.

    Infers Wyckoff/Weinstein stage from the combination of:
    - Price structure (base width, depth, tightening)
    - PCA score (buying pressure evidence)
    - EmScore (coiling mechanics)
    - Phase classification
    - Smart money verdict

    Returns AccumStage (NONE/1A/1B/1C/2A/2B), AccumStageLabel,
    AccumSequenceScore (0–100), AccumConfidence, AccumBarsInBase.
    """
    out = dict(
        AccumStage="NONE", AccumStageLabel="No base detected",
        AccumSequenceScore=0, AccumConfidence=0, AccumBarsInBase=0,
    )
    try:
        if df is None or len(df) < 30:
            return out
        cl  = df["Close"].values.astype(np.float64)
        hi  = df["High"].values.astype(np.float64)
        lo  = df["Low"].values.astype(np.float64)
        vol = df["Volume"].values.astype(np.float64)
        n   = len(cl)

        # ── Base detection ────────────────────────────────────────────────────
        # Find the longest recent stretch where price stayed within a range ≤ 25%
        # of price (Weinstein Stage 1 base = sideways consolidation)
        base_bars = 0
        base_depth_pct = 0.0
        base_range_tightening = False
        try:
            lb = min(120, n)
            cl_lb = cl[-lb:]
            hi_lb = hi[-lb:]
            lo_lb = lo[-lb:]
            # Scan backwards for base: find max contiguous stretch within 20% range
            for start in range(lb - 1, 0, -1):
                sub_hi  = float(np.max(hi_lb[start:]))
                sub_lo  = float(np.min(lo_lb[start:]))
                mid_px  = (sub_hi + sub_lo) / 2
                if mid_px > 0:
                    depth = (sub_hi - sub_lo) / mid_px * 100
                    if depth <= 25.0:
                        base_bars = lb - start
                        base_depth_pct = round(depth, 1)
                        # Tightening: first half range > second half range
                        mid = start + (lb - start) // 2
                        h1 = float(np.max(hi_lb[start:mid])) - float(np.min(lo_lb[start:mid]))
                        h2 = float(np.max(hi_lb[mid:])) - float(np.min(lo_lb[mid:]))
                        base_range_tightening = (h2 < h1 * 0.80)
                    else:
                        break
        except Exception:
            pass

        # ── Spring / failed breakdown detection ───────────────────────────────
        spring_detected = False
        try:
            if base_bars >= 10 and n >= 15:
                lb_spring = min(base_bars, n - 5)
                base_lo_support = float(np.min(lo[-lb_spring - 5:-5]))
                spring_cnt = sum(
                    1 for k in range(1, min(10, n))
                    if float(lo[-k]) < base_lo_support * 0.997
                    and float(cl[-k]) >= base_lo_support * 0.999
                )
                spring_detected = spring_cnt >= 1
        except Exception:
            pass

        # ── Price breakout from base ──────────────────────────────────────────
        early_markup = phase in (PHASE_BRK, PHASE_CONT, PHASE_ENTRY)
        continued_markup = phase in (PHASE_CONT, PHASE_BRK) and rs_line_high

        # ── Sequence scoring ──────────────────────────────────────────────────
        # Combine structural evidence with PCA + Em signals
        seq_score = 0
        confidence_signals = 0

        if base_bars >= 20:   seq_score += 25; confidence_signals += 1
        elif base_bars >= 10: seq_score += 15
        elif base_bars >= 5:  seq_score += 8

        if base_range_tightening: seq_score += 15; confidence_signals += 1
        if spring_detected:        seq_score += 20; confidence_signals += 1
        if pca_score >= 50:        seq_score += 15; confidence_signals += 1
        elif pca_score >= 35:      seq_score += 8
        if em_score >= 45:         seq_score += 10; confidence_signals += 1
        elif em_score >= 30:       seq_score += 5
        if smart_money_verdict in ("ACCUMULATING", "MARKUP_READY"):
            seq_score += 15; confidence_signals += 1
        elif smart_money_verdict == "ABSORBING":
            seq_score += 8

        seq_score = int(np.clip(seq_score, 0, 100))
        confidence = min(100, confidence_signals * 17)

        # ── Stage assignment ──────────────────────────────────────────────────
        if continued_markup and rs_line_high:
            stage = "2B"
        elif early_markup and (pca_score >= 45 or seq_score >= 60):
            stage = "2A"
        elif spring_detected and pca_score >= 40:
            stage = "1C"
        elif base_bars >= 15 and base_range_tightening and pca_score >= 30:
            stage = "1B"
        elif base_bars >= 8 and seq_score >= 25:
            stage = "1A"
        else:
            stage = "NONE"

        label = _ACCUM_STAGES.get(stage, (0, "Unknown"))[1]

        out.update(
            AccumStage          = stage,
            AccumStageLabel     = label,
            AccumSequenceScore  = seq_score,
            AccumConfidence     = confidence,
            AccumBarsInBase     = base_bars,
        )
    except Exception:
        pass
    return out

# ══════════════════════════════════════════════════════════════════════════════
# GAP-4 — MICROSTRUCTURE LOGIC
# Reconstructs intrabar order-flow proxies from OHLCV. Works for all modes.
# ══════════════════════════════════════════════════════════════════════════════

def compute_microstructure(df: pd.DataFrame, mode: str) -> dict:
    """
    Microstructure Score (0–100 MicroScore).

    Intrabar order-flow proxies reconstructed from OHLCV:
    1. Close Location Value  (20 pts) — where close sits in bar range (=buying pressure)
    2. Delta Proxy           (20 pts) — fraction of bar range used upward (up-close bias)
    3. Absorption Ratio      (20 pts) — high-vol bars with small net move (=supply absorbed)
    4. Wick Asymmetry        (20 pts) — lower wicks > upper wicks = buyers defending
    5. VWAP Micro-Deviation  (20 pts) — price hugging VWAP from above = strong demand

    Labels: STRONG_BUY_FLOW ≥75 · BUY_FLOW ≥55 · NEUTRAL_FLOW ≥35 ·
            SELL_FLOW ≥20 · STRONG_SELL_FLOW <20
    """
    out = dict(
        MicroScore=0.0, MicroLabel="NEUTRAL_FLOW",
        MicroDelta=0.0, MicroCLV=0.0,
        MicroAbsorption=0.0, MicroWickAsym=0.0, MicroVWAPDev=0.0,
    )
    try:
        if df is None or len(df) < 20:
            return out
        cl  = df["Close"].values.astype(np.float64)
        hi  = df["High"].values.astype(np.float64)
        lo  = df["Low"].values.astype(np.float64)
        vol = df["Volume"].values.astype(np.float64)
        op  = (df["Open"].values.astype(np.float64)
               if "Open" in df.columns else cl.copy())
        n   = len(cl)

        # ── 1. Close Location Value (0–20 pts) ────────────────────────────────
        # CLV = (close - low) / (high - low); averaged = where buyers left price
        clv_pts = 0.0
        try:
            lb_clv = min(20, n)
            rng    = hi[-lb_clv:] - lo[-lb_clv:]
            clv    = np.where(rng > 0, (cl[-lb_clv:] - lo[-lb_clv:]) / rng, 0.5)
            # Volume-weighted CLV
            vw_clv = float(np.sum(clv * vol[-lb_clv:]) / (np.sum(vol[-lb_clv:]) + 1e-10))
            if   vw_clv > 0.70: clv_pts = 20.0
            elif vw_clv > 0.60: clv_pts = 15.0
            elif vw_clv > 0.50: clv_pts = 10.0
            elif vw_clv > 0.40: clv_pts = 5.0
            elif vw_clv > 0.30: clv_pts = 2.0
        except Exception:
            pass

        # ── 2. Delta Proxy (0–20 pts) ─────────────────────────────────────────
        # Measures how much of the bar's range was used on the upside
        # Approximation: (close - open) / range = net directional usage
        delta_pts = 0.0
        try:
            lb_d  = min(20, n)
            rng_d = hi[-lb_d:] - lo[-lb_d:]
            delta = np.where(rng_d > 0, (cl[-lb_d:] - op[-lb_d:]) / rng_d, 0.0)
            # Volume-weighted mean delta
            vw_delta = float(np.sum(delta * vol[-lb_d:]) / (np.sum(vol[-lb_d:]) + 1e-10))
            if   vw_delta > 0.25: delta_pts = 20.0
            elif vw_delta > 0.15: delta_pts = 15.0
            elif vw_delta > 0.05: delta_pts = 10.0
            elif vw_delta > 0:    delta_pts = 5.0
            elif vw_delta > -0.05:delta_pts = 2.0
        except Exception:
            pass

        # ── 3. Absorption Ratio (0–20 pts) ────────────────────────────────────
        # High-vol bars with small price move relative to vol = supply being absorbed
        abs_pts = 0.0
        try:
            avg_vol_ = float(np.mean(vol[-21:-1])) if n >= 22 else float(np.mean(vol))
            avg_rng_ = float(np.mean(hi[-21:-1] - lo[-21:-1])) if n >= 22 else float(np.mean(hi - lo))
            if avg_vol_ > 0 and avg_rng_ > 0:
                lb_a = min(15, n - 1)
                absorption_score = 0.0
                for k in range(1, lb_a + 1):
                    v_k   = float(vol[-k])
                    rng_k = float(hi[-k] - lo[-k])
                    # Volume efficiency ratio: vol per unit of range
                    if rng_k > 0:
                        eff = v_k / rng_k
                        avg_eff = avg_vol_ / (avg_rng_ + 1e-10)
                        # High efficiency + close in upper half = absorption
                        clv_k = (float(cl[-k]) - float(lo[-k])) / rng_k
                        if eff > avg_eff * 1.5 and clv_k > 0.50:
                            absorption_score += 2.0
                        elif eff > avg_eff * 1.2 and clv_k > 0.45:
                            absorption_score += 1.0
                abs_pts = min(20.0, absorption_score)
        except Exception:
            pass

        # ── 4. Wick Asymmetry (0–20 pts) ─────────────────────────────────────
        # Lower wicks > upper wicks = sellers tried to push down, buyers recovered
        wick_pts = 0.0
        try:
            lb_w   = min(20, n)
            upper_wicks = hi[-lb_w:] - np.maximum(cl[-lb_w:], op[-lb_w:])
            lower_wicks = np.minimum(cl[-lb_w:], op[-lb_w:]) - lo[-lb_w:]
            # Volume-weight each wick
            vw_upper = float(np.sum(upper_wicks * vol[-lb_w:]) / (np.sum(vol[-lb_w:]) + 1e-10))
            vw_lower = float(np.sum(lower_wicks * vol[-lb_w:]) / (np.sum(vol[-lb_w:]) + 1e-10))
            ratio = vw_lower / (vw_upper + 1e-10)
            if   ratio > 2.5: wick_pts = 20.0
            elif ratio > 1.8: wick_pts = 15.0
            elif ratio > 1.3: wick_pts = 10.0
            elif ratio > 1.0: wick_pts = 6.0
            elif ratio > 0.7: wick_pts = 2.0
        except Exception:
            pass

        # ── 5. VWAP Micro-Deviation (0–20 pts) ────────────────────────────────
        # How consistently price closes above rolling VWAP over recent bars
        vwap_pts = 0.0
        try:
            lb_v = min(20, n)
            typ  = (hi[-lb_v:] + lo[-lb_v:] + cl[-lb_v:]) / 3.0
            cum_vol  = np.cumsum(vol[-lb_v:])
            cum_tpv  = np.cumsum(typ * vol[-lb_v:])
            vwap_arr = cum_tpv / (cum_vol + 1e-10)
            # Count bars where close > rolling VWAP
            above_vwap = np.sum(cl[-lb_v:] > vwap_arr)
            pct_above  = above_vwap / lb_v
            # Also check: is close currently above VWAP?
            above_now  = float(cl[-1]) > float(vwap_arr[-1])
            if   pct_above > 0.80 and above_now: vwap_pts = 20.0
            elif pct_above > 0.65 and above_now: vwap_pts = 15.0
            elif pct_above > 0.55 and above_now: vwap_pts = 10.0
            elif pct_above > 0.45 and above_now: vwap_pts = 5.0
            elif above_now:                       vwap_pts = 3.0
        except Exception:
            pass

        total = round(float(np.clip(
            clv_pts + delta_pts + abs_pts + wick_pts + vwap_pts,
            0, 100
        )), 1)

        if   total >= 75: label = "STRONG_BUY_FLOW"
        elif total >= 55: label = "BUY_FLOW"
        elif total >= 35: label = "NEUTRAL_FLOW"
        elif total >= 20: label = "SELL_FLOW"
        else:             label = "STRONG_SELL_FLOW"

        out.update(
            MicroScore      = total,
            MicroLabel      = label,
            MicroCLV        = round(clv_pts,   1),
            MicroDelta      = round(delta_pts,  1),
            MicroAbsorption = round(abs_pts,    1),
            MicroWickAsym   = round(wick_pts,   1),
            MicroVWAPDev    = round(vwap_pts,   1),
        )
    except Exception:
        pass
    return out

# ══════════════════════════════════════════════════════════════════════════════
# v15.1 PATTERN HELPERS + v15.2 FIXES (closed-candle, pivot tolerance)
# ══════════════════════════════════════════════════════════════════════════════

# FIX-A: pivot tolerance multipliers per mode
_PIVOT_TOL = {"Intraday": 0.25, "Swing": 0.15, "Positional": 0.10}


def category_score(*,
                   trend_up, ema_stack, fresh_cross, htf_up, market_bullish, e_fast_gt_slow,
                   rsi, mom1, mom3, mom6, mom1_th, mom3_th, mom6_th,
                   phase, in_golden, near_e127, near_e161, norm_bull_raw,
                   rs_rank, c_gt_hh, c_near_hh,
                   vol_ratio, vol_avg_gt_zero, adx_val,
                   squeeze, vc_ratio, ext_penalty,
                   regime_bearish,
                   # Engine 1 — MTF sync (optional, safe default = neutral)
                   mtf_sync_score: float = 50.0,
                   # Engine 2 — Institutional volume (optional, safe default = neutral)
                   inst_score: float = 50.0,
                   # Engine 3 — Harmonics (optional)
                   harmonic_score: int = 0,
                   # Engine 5 — Candle structure (optional)
                   candle_score: int = 0,
                   # Engine 4 — Adaptive regime weights (None → use static _CAT_W)
                   regime_weights: dict | None = None) -> dict:

    W = regime_weights if regime_weights is not None else _CAT_W

    # TREND — MTF sync adds ±10 raw points (centred at 50)
    t = (40 if trend_up else 0) \
      + (20 if ema_stack else (10 if e_fast_gt_slow else 0)) \
      + (15 if htf_up else 0) \
      + (15 if market_bullish else 0) \
      + (10 if fresh_cross else 0)
    t = max(0.0, t + (mtf_sync_score - 50.0) / 50.0 * 10.0)
    cat_T = min(W["TREND"], t / 100 * W["TREND"])

    # MOMENTUM
    m = (40 if rsi>=70 else 35 if rsi>=65 else 25 if rsi>=60 else 18 if rsi>=55
         else 10 if rsi>=50 else 0 if rsi>=40 else -10)
    m += (25 if mom1>mom1_th else 12 if mom1>0 else -5)
    m += (20 if mom3>mom3_th else 10 if mom3>0 else 0)
    m += (15 if mom6>mom6_th else 5 if mom6>0 else 0)
    cat_M = min(W["MOMENTUM"], max(0.0, m) / 100 * W["MOMENTUM"])

    # STRUCTURE — harmonic patterns add up to +12 raw
    s = float(_PHASE_RAW.get(phase, 10))
    s += (20 if c_gt_hh else (10 if c_near_hh else 0))
    s += (20 if in_golden else 0)
    s += (-25 if near_e127 else (-35 if near_e161 else 0))
    s += (15 if rs_rank>=80 else (5 if rs_rank>=60 else (-10 if rs_rank<30 else 0)))
    s += harmonic_score * 0.15   # quality*0.8 → max ~64; ×0.15 → ≤9.6 raw pts
    cat_S = min(W["STRUCTURE"], max(0.0, s) / 120 * W["STRUCTURE"])

    # VOLUME — institutional score replaces static ADX-only centre
    v = 0.0
    if vol_avg_gt_zero:
        v += (50 if vol_ratio>=1.5 else 35 if vol_ratio>=1.2 else 20 if vol_ratio>=1.0 else -5)
    v += (35 if adx_val>=30 else 20 if adx_val>=20 else 8 if adx_val>=15 else -8)
    v += (inst_score - 50.0) / 50.0 * 15.0   # institutional: ±15 raw
    cat_V = min(W["VOLUME"], max(0.0, v) / 100 * W["VOLUME"])

    # QUALITY — candle structure adds ±10 raw points
    q = (25 if squeeze else 0) + (25 if vc_ratio<0.75 else (12 if vc_ratio<0.90 else 0))
    q += max(0.0, 40.0 + ext_penalty)
    q += candle_score   # −10 to +10
    cat_Q = min(W["QUALITY"], q / 100 * W["QUALITY"])

    raw = cat_T + cat_M + cat_S + cat_V + cat_Q
    if regime_bearish:
        raw *= 0.85
    return dict(norm_bull=round(min(100.0, max(0.0, raw)), 1),
                cat_T=round(cat_T,2), cat_M=round(cat_M,2),
                cat_S=round(cat_S,2), cat_V=round(cat_V,2), cat_Q=round(cat_Q,2))

# ══════════════════════════════════════════════════════════════════════════════
# CORE SCORING — v14 logic + SPEED-10 new indicators
# ══════════════════════════════════════════════════════════════════════════════

def score_stock(df, nifty_close, mode="Swing", daily_close=None,
                market_bullish=True, vix_val=None, min_liquidity_cr=LIQUIDITY_MIN_CR,
                sym=None, htf_up=True, rs_rank=50,
                phase_history_snapshot=None, mtf_prefetched=None):
    try:
        cfg   = MODE_CFG[mode]
        close = df["Close"]; volume = df["Volume"]; n = len(close)
        if n < 50: return None

        liq_ok, avg_cr = liquidity_ok(df, min_liquidity_cr, mode=mode)

        c        = float(close.iloc[-1])
        prev     = float(close.iloc[-2])
        e_fast_s = ema(close, cfg["ema_fast"])
        e_slow_s = ema(close, cfg["ema_slow"])
        e_fast   = float(e_fast_s.iloc[-1])
        e_slow   = float(e_slow_s.iloc[-1])
        e200_s   = ema(close, 200)
        e200     = float(e200_s.iloc[-1]) if n >= 200 else None
        atr_s    = atr_series(df)
        atr_val  = float(atr_s.iloc[-1])
        atr_mean = float(atr_s.rolling(20).mean().iloc[-1])
        chg      = round(((c-prev)/prev)*100, 2)
        hh       = float(close.iloc[-11:-1].max())

        if len(df) >= 2:
            try:    delta_min = (df.index[1]-df.index[0]).total_seconds()/60
            except: delta_min = 1440
        else:
            delta_min = 1440

        if delta_min <= 5:     bars_per_day = 75
        elif delta_min <= 15:  bars_per_day = 25
        elif delta_min <= 30:  bars_per_day = 13
        elif delta_min < 240:  bars_per_day = 7
        else:                  bars_per_day = 1

        if mode == "Intraday" and bars_per_day > 1:
            vol_avg = _intraday_vol_avg(volume, bars_per_day)
        else:
            vol_avg = float(volume.rolling(20).mean().iloc[-1])

        v           = float(volume.iloc[-1])
        above_ema50 = c > float(ema(close, 50).iloc[-1])

        rs_raw = 0.0
        if n >= 6 and len(nifty_close) >= 6:
            rs_raw = ((c-float(close.iloc[-6]))/float(close.iloc[-6]) -
                      (float(nifty_close.iloc[-1])-float(nifty_close.iloc[-6]))/
                      float(nifty_close.iloc[-6]))*100

        trend_up     = (e200 is None or c > e200) and c > e_fast and e_fast > e_slow
        trend_down   = (e200 is None or c < e200) and c < e_fast and e_fast < e_slow
        trend_strong = c > e_fast and e_fast > e_slow
        ema_stack    = (e200 is not None) and (c > e200) and (e_fast > e_slow) and (e_fast > e200)

        fresh_cross = False
        if n >= 6 and e_fast > e_slow:
            lookback_cross = min(5, n-1)
            for k in range(1, lookback_cross+1):
                ef_curr = float(e_fast_s.iloc[-k]); es_curr = float(e_slow_s.iloc[-k])
                ef_prev = float(e_fast_s.iloc[-(k+1)]); es_prev = float(e_slow_s.iloc[-(k+1)])
                if ef_curr > es_curr and ef_prev <= es_prev:
                    fresh_cross = True; break

        ema_cross_bonus = 8 if fresh_cross else (4 if e_fast > e_slow else 0)

        mom_src = (daily_close if (mode=="Intraday" and daily_close is not None
                                    and len(daily_close)>=21) else close)
        mom_n   = len(mom_src)
        mom1 = (c-float(mom_src.iloc[-21]))  /float(mom_src.iloc[-21])*100  if mom_n>=21  else 0
        mom3 = (c-float(mom_src.iloc[-63]))  /float(mom_src.iloc[-63])*100  if mom_n>=63  else 0
        mom6 = (c-float(mom_src.iloc[-126])) /float(mom_src.iloc[-126])*100 if mom_n>=126 else 0
        strong_htf = mom1>cfg["mom1_th"] and mom3>cfg["mom3_th"] and mom6>cfg["mom6_th"]

        sw_hi, sw_lo, fib, fib_rng = fib_levels(df, lookback=30)
        prox      = atr_val*0.3
        in_golden = bool(fib and c>=fib["618"]-prox and c<=fib["500"]+prox)
        near_e127 = bool(fib and abs(c-fib["ext127"]) < prox)
        near_e161 = bool(fib and abs(c-fib["ext161"]) < prox)

        VDU_VOL_RATIO=0.70; VDU_RANGE_MULT=0.80
        vdu_vol_dry=False; vdu_coil=False
        if n >= 20 and vol_avg > 0:
            recent_vols = [float(volume.iloc[k]) for k in [-3,-2,-1]]
            vdu_vol_dry = all(vv < vol_avg*VDU_VOL_RATIO for vv in recent_vols)
        if n >= 5:
            recent_hi = float(df["High"].iloc[-5:].max())
            recent_lo = float(df["Low"].iloc[-5:].min())
            vdu_coil  = (recent_hi-recent_lo) < atr_val*VDU_RANGE_MULT
        vdu_setup  = bool(trend_up and vdu_vol_dry and vdu_coil)
        qualified  = strong_htf and trend_strong

        rsi_series = rsi(close, cfg["rsi_len"])
        ext_flags, ext_penalty, ext_labels, ext_n = detect_exhaustion(
            close=close, high=df["High"], low=df["Low"], volume=volume,
            rsi_series=rsi_series, e_fast_s=e_fast_s, atr_s=atr_s, atr_mean=atr_mean,
            c=c, v=v, vol_avg=vol_avg, mode=mode, vix_val=vix_val,
        )
        r = float(rsi_series.iloc[-1])

        # ── SPEED-10: NEW STAGE-B INDICATORS ──────────────────────────────
        adx_val   = _compute_adx(df)
        squeeze   = _compute_squeeze(df)
        sqz_bars  = _count_squeeze_bars(df) if squeeze else 0
        sqz_depth = _sqz_depth(df)          if squeeze else 1.0
        vol_ratio = _compute_vol_contraction(df)

        # ── NEW ENGINES (v15.4) ────────────────────────────────────────────
        # Engine 4 first — its weights feed both category_score calls below
        regime_bearish = not market_bullish
        _regime_key, _regime_label, _regime_w = classify_regime(
            market_bullish, adx_val, vix_val)

        # Engine 2: institutional volume
        _inst = analyze_institutional_volume(df, mode)

        # Engine 3 & 5: pattern engines are SKIPPED in the main scan for performance.
        # At NSE-500 scale these are the marginal compute cost on 300+ survivors.
        # Both engines run post-scan via enrich_with_patterns() on shortlisted stocks only.
        _harm = {
            "detected": False, "pattern": "", "direction": "NEUTRAL",
            "quality": 0, "completion_zone": False, "harmonic_score": 0,
        }
        _candle = {
            "candle_score": 0, "candle_signal": "", "patterns": [],
            "nr7": False, "inside_bar": False,
        }

        # Engine 1: MTF sync — inject base TF df so all three TFs are present
        _mtf_data = dict(mtf_prefetched or {})
        _base_tf   = MODE_CFG[mode]["interval"]
        if _base_tf not in _mtf_data:
            _mtf_data[_base_tf] = df
        _mtf = compute_mtf_sync(sym or "", mode, prefetched=_mtf_data)
        # ── FIX-C: category-based scoring — placeholder phase ──────────────
        _cat = category_score(
            trend_up       = trend_up,
            ema_stack      = ema_stack,
            fresh_cross    = fresh_cross,
            htf_up         = htf_up,
            market_bullish = market_bullish,
            e_fast_gt_slow = (e_fast > e_slow),
            rsi            = r,
            mom1=mom1, mom3=mom3, mom6=mom6,
            mom1_th=cfg["mom1_th"], mom3_th=cfg["mom3_th"], mom6_th=cfg["mom6_th"],
            phase          = PHASE_IDLE,        # placeholder; overwritten after detect_phase
            in_golden      = in_golden,
            near_e127      = near_e127,
            near_e161      = near_e161,
            norm_bull_raw  = 50.0,
            rs_rank        = rs_rank,
            c_gt_hh        = (c > hh),
            c_near_hh      = (c > hh*0.98),
            vol_ratio      = (v/vol_avg) if vol_avg > 0 else 1.0,
            vol_avg_gt_zero= vol_avg > 0,
            adx_val        = adx_val,
            squeeze        = squeeze,
            vc_ratio       = vol_ratio,
            ext_penalty    = ext_penalty,
            regime_bearish = regime_bearish,
            mtf_sync_score = _mtf["sync_score"],
            inst_score     = _inst["inst_score"],
            harmonic_score = _harm["harmonic_score"],
            candle_score   = _candle["candle_score"],
            regime_weights = _regime_w,
        )
        norm_bull = _cat["norm_bull"]
        raw_score = int(norm_bull)
        score_th  = float(cfg["score_th"])

        act           = action_label(norm_bull)
        vol_confirmed = v > vol_avg*1.2

        phase, entry_price, setup_type = detect_phase_and_entry(
            df, mode, c=c, e_fast_s=e_fast_s, e_slow_s=e_slow_s,
            atr_s=atr_s, atr_val=atr_val, atr_mean=atr_mean,
            v=v, vol_avg=vol_avg, fib=fib, sw_hi=sw_hi, sw_lo=sw_lo,
            in_golden=in_golden, near_e127=near_e127, near_e161=near_e161,
            norm_bull=norm_bull, trend_up=trend_up, trend_down=trend_down,
            trend_strong=trend_strong, score_th=score_th, vdu_setup=vdu_setup,
            htf_up=htf_up, regime_bearish=regime_bearish, vix_val=vix_val,
        )

        # Re-score with the real phase now known
        _cat2 = category_score(
            trend_up=trend_up, ema_stack=ema_stack, fresh_cross=fresh_cross,
            htf_up=htf_up, market_bullish=market_bullish, e_fast_gt_slow=(e_fast>e_slow),
            rsi=r, mom1=mom1, mom3=mom3, mom6=mom6,
            mom1_th=cfg["mom1_th"], mom3_th=cfg["mom3_th"], mom6_th=cfg["mom6_th"],
            phase=phase, in_golden=in_golden, near_e127=near_e127, near_e161=near_e161,
            norm_bull_raw=norm_bull, rs_rank=rs_rank,
            c_gt_hh=(c>hh), c_near_hh=(c>hh*0.98),
            vol_ratio=(v/vol_avg) if vol_avg>0 else 1.0,
            vol_avg_gt_zero=vol_avg>0, adx_val=adx_val, squeeze=squeeze,
            vc_ratio=vol_ratio, ext_penalty=ext_penalty, regime_bearish=regime_bearish,
            mtf_sync_score=_mtf["sync_score"],
            inst_score=_inst["inst_score"],
            harmonic_score=_harm["harmonic_score"],
            candle_score=_candle["candle_score"],
            regime_weights=_regime_w,
        )
        norm_bull = _cat2["norm_bull"]
        raw_score = int(norm_bull)

        # ADX gate: don't declare BREAKOUT if ADX weak (no trend strength)
        if phase == PHASE_BRK and adx_val < 18:
            phase = PHASE_ENTRY

        phase, _ = ext_phase_override(phase, ext_flags, ext_n, mode)
        act       = ext_action_cap(act, ext_n, vix_val)

        phase_bonus = 0
        if sym and phase_history_snapshot:
            history = phase_history_snapshot.get(sym, [])
            if len(history) >= 3:
                last3 = [h[1] for h in history[-3:]]
                progressions = [
                    [PHASE_SETUP,PHASE_ENTRY,PHASE_CONT],
                    [PHASE_ENTRY,PHASE_CONT,PHASE_BRK],
                    [PHASE_SETUP,PHASE_ENTRY,PHASE_BRK],
                ]
                phase_bonus = 5 if last3 in progressions else 0

        confidence = compute_confidence(
            norm_bull=norm_bull, phase=phase, trend_up=trend_up,
            trend_strong=trend_strong, vol_confirmed=vol_confirmed,
            ema_stack=ema_stack, htf_aligned=htf_up,
            regime_bullish=market_bullish, ext_n=ext_n, vix_val=vix_val,
            phase_bonus=phase_bonus, rs_rank=rs_rank,
        )

        ltp   = round(c, 2)
        entry = entry_price if entry_price else ltp

        # ── EXPLICIT STOP-LOSS (compute_stop_loss owns SL logic) ──────────
        # Detect if squeeze was active — use squeeze SL rule when appropriate
        _sl_setup = setup_type
        if _sl_setup not in ("fib", "breakout", "support") and squeeze:
            _sl_setup = "squeeze"
        sl = compute_stop_loss(
            entry      = entry,
            atr_val    = atr_val,
            setup_type = _sl_setup,
            mode       = mode,
            fib        = fib if fib else None,
            sw_lo      = sw_lo,
        )

        # ── DYNAMIC TARGETS (scale by EM energy, ExtN, MicroScore) ─────────
        _em_lbl_for_tgt = "QUIET"   # will be filled by compute_emerging_score below
        # Use synchronous values available now; targets are re-used as-is
        # (dynamic targets require em_label — computed below — but a two-pass
        #  would be expensive; we instead pass the raw EmScore proxy here and
        #  re-derive the label in the post-hoc enrichment pass in run_scan)
        _em_pts_approx = float(compute_emerging_score(df, mode, nifty_close, rs_rank).get("EmScore", 0.0))
        _em_lbl_for_tgt = (
            "IGNITING" if _em_pts_approx >= 65 else
            "BUILDING" if _em_pts_approx >= 50 else
            "COILING"  if _em_pts_approx >= 35 else
            "LATENT"   if _em_pts_approx >= 20 else "QUIET"
        )
        _micro_for_tgt = compute_microstructure(df, mode).get("MicroScore", 50.0)
        t1, t2, t3 = compute_dynamic_targets(
            entry          = entry,
            sl             = sl,
            atr_val        = atr_val,
            setup_type     = setup_type,
            fib            = fib if fib else None,
            sw_hi          = sw_hi,
            sw_lo          = sw_lo,
            em_label       = _em_lbl_for_tgt,
            ext_n          = ext_n,
            micro_score    = _micro_for_tgt,
            vix_val        = vix_val,
            regime_bearish = regime_bearish,
        )
        sl_exp = vix_target_mult(vix_val)[3]
        if sl_exp > 1.0:
            sl = round(entry - (entry - sl) * sl_exp, 2)

        return {
            "Score":       round(norm_bull,1), "RawBull":raw_score,
            "Action":      act, "Phase":phase, "Setup":setup_type,
            "Confidence":  confidence, "%Change":chg,
            "LTP":         ltp, "Entry":entry, "SL":sl,
            "T1":t1,"T2":t2,"T3":t3,
            "InGolden":    in_golden, "VDU":vdu_setup,
            "AboveEMA50":  above_ema50, "AvgTradedCr":avg_cr,
            "LiquidityOK": liq_ok, "RSI":round(r,1),
            "RS":          round(rs_raw,2), "RS_Rank":rs_rank,
            "ExtN":        ext_n, "ExtLabels":ext_labels,
            "ExtFlags":    ext_flags, "HTFUp":htf_up,
            "EMAStack":    ema_stack, "VolConf":vol_confirmed,
            "FreshCross":  fresh_cross, "ATR":round(atr_val,2),
            "ATR_Mean":    round(atr_mean,2), "PhaseBonus":phase_bonus,
            "EMA200":      round(e200, 2) if e200 else None,
            "BreadthGated":False, "Mom1":round(mom1,2), "Mom3":round(mom3,2),
            "TrendUp":     trend_up, "TrendDown":trend_down,
            # v15 speed-10
            "ADX":         round(adx_val,1), "Squeeze":squeeze,
            "SqzBars":     sqz_bars, "SqzDepth":round(sqz_depth,3),
            "VolRatio":    round(vol_ratio,2),
            # v15.3 category scores
            "CatT":_cat2["cat_T"],"CatM":_cat2["cat_M"],"CatS":_cat2["cat_S"],
            "CatV":_cat2["cat_V"],"CatQ":_cat2["cat_Q"],
            # v15.1/15.2 pattern keys — populated by enrich_with_patterns after Stage-B
            "Patterns":{},"VCP":False,"VCPGrade":"NONE","AVWAP":None,
            "AVWAPAbove":False,"FibQuality":0,"FibGrade":"POOR",
            "VolDryup":False,"VDUIntensity":0,"RVolPct":50.0,"RVolLabel":"NORMAL",
            "DarvasIn":False,"DarvasBrk":False,"DarvasTop":0.0,
            "_detected_phase": phase,
            # ── v15.4: Five new engines ───────────────────────────────────────
            # Engine 1 — MTF sync
            "MTFScore":   _mtf["sync_score"],
            "MTFLabel":   _mtf["mtf_label"],
            "MTFAligned": _mtf["aligned"],
            "MTFDiverge": _mtf["divergence"],
            "MTFTFScores":_mtf["tf_scores"],
            # Engine 2 — Institutional volume
            "InstScore":  _inst["inst_score"],
            "InstVerdict":_inst["verdict"],
            "InstLabel":  _inst["inst_label"],
            "InstEVR":    _inst["effort_vs_result"],
            "InstCMF":    _inst["cmf"],
            "InstOBV":    _inst["obv_trend"],
            "InstBlocks": _inst["block_days"],
            # Engine 3 — Harmonic / ABCD
            "HarmonicDetected": _harm["detected"],
            "HarmonicPattern":  _harm["pattern"],
            "HarmonicDir":      _harm["direction"],
            "HarmonicQuality":  _harm["quality"],
            "HarmonicZone":     _harm["completion_zone"],
            "HarmonicScore":    _harm["harmonic_score"],
            # Engine 4 — Adaptive regime
            "RegimeKey":    _regime_key,
            "RegimeLabel":  _regime_label,
            "RegimeWeights":_regime_w,
            # Engine 5 — Candle structure
            "CandleScore":  _candle["candle_score"],
            "CandleSignal": _candle["candle_signal"],
            "CandlePatterns":_candle["patterns"],
            "NR7":          _candle["nr7"],
            "InsideBar":    _candle["inside_bar"],
            # ── v15.5: Emerging Momentum Score (7-component leading indicator) ──
            **compute_emerging_score(df, mode, nifty_close, rs_rank),
            # ── v15.6: Pre-Confirmation Accumulation (7-signal buying-pressure layer) ──
            **compute_preconfirmation_accumulation(df, mode),
            # ── v15.7: Smart Money Behavior Model ────────────────────────────
            **compute_smart_money_model(
                df, mode,
                pca_score  = 0.0,     # filled post-hoc below
                inst_score = _inst["inst_score"],
                obv_trend  = _inst["obv_trend"],
            ),
            # ── v15.7: Accumulation Sequencing ───────────────────────────────
            **compute_accumulation_sequence(
                df, mode,
                pca_score            = 0.0,   # filled post-hoc below
                em_score             = 0.0,   # filled post-hoc below
                phase                = phase,
                smart_money_verdict  = "NEUTRAL",  # filled post-hoc below
                rs_line_high         = False,       # filled post-hoc below
            ),
            # ── v15.7: Microstructure Logic ───────────────────────────────────
            **compute_microstructure(df, mode),
            # ── v15.7: Relative Leadership Intelligence ───────────────────────
            **compute_rs_leadership(
                close            = close,
                nifty_close      = nifty_close,
                rs_rank          = rs_rank,
                sector_avg_score = 50.0,  # enriched in run_scan (like EmSectorMom)
                stock_score      = norm_bull,
            ),
        }
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# BREADTH ENGINE (unchanged from v14)
# ══════════════════════════════════════════════════════════════════════════════

