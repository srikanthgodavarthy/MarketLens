"""
indicators.py — Vectorised batch indicators and per-symbol technical helpers.
"""
import numpy as np
import pandas as pd
from typing import Optional

from config import MODE_CFG, LIQUIDITY_MIN_CR

def _ema_np(arr: np.ndarray, span: int) -> np.ndarray:
    """EMA on 1-D array."""
    alpha  = 2.0 / (span + 1)
    result = np.empty_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    return result

def _rma_np(arr: np.ndarray, period: int) -> np.ndarray:
    """Wilder's Moving Average (RMA) on 1-D array — alpha = 1/period.
    Used for ADX/DI smoothing to match charting platform values."""
    alpha  = 1.0 / period
    result = np.empty_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    return result

def _rma_batch(matrix: np.ndarray, period: int) -> np.ndarray:
    """Wilder's Moving Average on (N, T) matrix — alpha = 1/period."""
    alpha  = 1.0 / period
    result = np.empty_like(matrix)
    result[:, 0] = matrix[:, 0]
    beta = 1 - alpha
    for t in range(1, matrix.shape[1]):
        result[:, t] = alpha * matrix[:, t] + beta * result[:, t - 1]
    return result

def _ema_batch(matrix: np.ndarray, span: int) -> np.ndarray:
    """EMA on (N, T) matrix → (N, T). Uses numba if available, else loops."""
    N, T   = matrix.shape
    alpha  = 2.0 / (span + 1)
    result = np.empty_like(matrix)
    result[:, 0] = matrix[:, 0]
    beta = 1 - alpha
    for t in range(1, T):
        result[:, t] = alpha * matrix[:, t] + beta * result[:, t-1]
    return result

def _rsi_batch(close_matrix: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI on (N, T) matrix → (N, T)."""
    N, T   = close_matrix.shape
    diff   = np.diff(close_matrix, axis=1, prepend=close_matrix[:, :1])
    gain   = np.where(diff > 0, diff, 0.0)
    loss   = np.where(diff < 0, -diff, 0.0)
    avg_g  = _ema_batch(gain, period)
    avg_l  = _ema_batch(loss, period)
    rs     = np.where(avg_l == 0, 100.0, avg_g / (avg_l + 1e-10))
    return 100 - (100 / (1 + rs))

def _atr_batch(high: np.ndarray, low: np.ndarray,
               close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR on (N, T) matrices → (N, T)."""
    prev_close = np.roll(close, 1, axis=1)
    prev_close[:, 0] = close[:, 0]
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return _ema_batch(tr, period)

def _adx_batch(high: np.ndarray, low: np.ndarray,
               close: np.ndarray, period: int = 14) -> np.ndarray:
    """ADX on (N, T) → returns ADX values (N, T).
    Uses Wilder's RMA (alpha=1/period) to match standard charting platform values."""
    prev_high  = np.roll(high,  1, axis=1); prev_high[:,  0] = high[:,  0]
    prev_low   = np.roll(low,   1, axis=1); prev_low[:,   0] = low[:,   0]
    prev_close = np.roll(close, 1, axis=1); prev_close[:, 0] = close[:, 0]

    up_move   = high  - prev_high
    down_move = prev_low - low
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = np.maximum(high - low,
         np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

    atr_s     = _rma_batch(tr,        period)
    plus_di   = 100 * _rma_batch(plus_dm,  period) / (atr_s + 1e-10)
    minus_di  = 100 * _rma_batch(minus_dm, period) / (atr_s + 1e-10)
    dx        = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx       = _rma_batch(dx, period)
    return adx

def _bb_squeeze_batch(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                      period: int = 20, bb_mult: float = 2.0,
                      kc_mult: float = 1.5) -> np.ndarray:
    """
    Keltner/BB Squeeze detector.
    Returns boolean (N, T) — True = squeeze (BB inside KC).
    """
    N, T = close.shape
    # Rolling mean + std (approx with EMA for speed)
    mid    = _ema_batch(close, period)
    # Approximate rolling std via EMA of squared deviations
    dev2   = _ema_batch((close - mid) ** 2, period)
    std    = np.sqrt(np.maximum(dev2, 0))
    bb_up  = mid + bb_mult * std
    bb_lo  = mid - bb_mult * std
    # Keltner channels via ATR
    atr_k  = _atr_batch(high, low, close, period)
    kc_up  = mid + kc_mult * atr_k
    kc_lo  = mid - kc_mult * atr_k
    # Squeeze = BB inside KC
    squeeze = (bb_up <= kc_up) & (bb_lo >= kc_lo)
    return squeeze

def _vol_contraction_batch(atr_matrix: np.ndarray) -> np.ndarray:
    """
    Volatility contraction ratio: ATR_5 / ATR_20.
    Values < 0.75 indicate compression.
    Returns (N,) array of latest ratios.
    """
    atr_short = _ema_batch(atr_matrix, 5)
    atr_long  = _ema_batch(atr_matrix, 20)
    ratio = atr_short[:, -1] / (atr_long[:, -1] + 1e-10)
    return ratio

# ══════════════════════════════════════════════════════════════════════════════
# SPEED-2: TWO-STAGE SCAN
#  Stage-A: fast pre-filter (price, volume, EMA) — eliminates ~65% of symbols
#  Stage-B: full engine on survivors
# ══════════════════════════════════════════════════════════════════════════════

def stage_a_prefilter(data: dict[str, pd.DataFrame],
                       mode: str, min_bars: int = 30) -> list[str]:
    """
    SPEED-2 TIER-1 FILTER — vectorized, runs on ALL valid symbols.

    Old behaviour: hard EMA-alignment gate dropped ~65% of symbols, silently
    discarding accumulating and emerging stocks that are *supposed* to be below
    their EMAs (that's the whole point of base-building).

    New behaviour: two-tier approach.
      Tier-1 (this function): only removes genuinely dead stocks:
        • Price ≤ 10 (penny / halted)
        • Volume = 0 on the last bar (no trading activity at all)
        • Extreme downtrend: close < 50% of 52-week high (deeply broken stocks)
          This is a wide enough gate that accumulating stocks pass easily.

      Tier-2 (score_stock): full engine scoring — nothing is thrown away before
        the engines get a chance to detect accumulation or emerging coiling.

    Speed improvement without Stage-A:
      The scoring bottleneck is not the number of stocks scored but the number
      of external HTF / MTF fetches. Those are still only done for survivors of
      this wider Tier-1 gate, keeping network cost under control.
      The score_stock function itself takes <2ms per stock on cached data, so
      scoring 500 vs 180 adds <700ms on a 32-worker pool — acceptable.
    """
    cfg      = MODE_CFG[mode]
    ef_span  = cfg["ema_fast"]
    es_span  = cfg["ema_slow"]

    symbols  = [s for s, df in data.items()
                if df is not None and not df.empty and len(df) >= min_bars]
    if not symbols:
        return []

    # Vectorized batch: build close/volume matrices once
    max_len = max(len(data[s]) for s in symbols)
    closes  = np.zeros((len(symbols), max_len), dtype=np.float32)
    vols    = np.zeros((len(symbols), max_len), dtype=np.float32)

    for i, sym in enumerate(symbols):
        cl = data[sym]["Close"].values.astype(np.float32)
        hi = data[sym]["High"].values.astype(np.float32)
        n  = len(cl)
        closes[i, max_len - n:] = cl
        vols[i,   max_len - n:] = data[sym]["Volume"].values.astype(np.float32)
        if n < max_len:
            closes[i, :max_len - n] = cl[0]
            vols[i,   :max_len - n] = vols[i, max_len - n]

    c_last  = closes[:, -1]
    v_last  = vols[:, -1]

    # 52-week high (last 252 bars at most)
    lookback = min(252, max_len)
    hi_52w   = closes[:, -lookback:].max(axis=1)

    # ── TIER-1 GATES (dead stock removal only) ─────────────────────────────
    # Gate 1: penny/halted stocks
    price_ok   = c_last > 10.0
    # Gate 2: zero volume — stock has no activity at all
    vol_ok     = v_last > 0
    # Gate 3: deeply broken (< 40% of 52w high) — not accumulating, just broken
    #         Use 40% not 50% to keep stocks in 40–50% drawdown (deep accumulation)
    not_broken = c_last >= (hi_52w * 0.40)

    passed  = price_ok & vol_ok & not_broken
    survivors = [sym for sym, ok in zip(symbols, passed.tolist()) if ok]
    return survivors

# ══════════════════════════════════════════════════════════════════════════════
# MATH HELPERS (unchanged from v14)
# ══════════════════════════════════════════════════════════════════════════════

def to_nse(sym):
    sym = sym.strip().upper()
    return sym if sym.endswith(".NS") else sym + ".NS"

def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(alpha=1/p, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/p, adjust=False).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def atr_series(df, p=14):
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        (hi - lo),
        (hi - cl.shift()).abs(),
        (lo - cl.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()

def fib_levels(df, lookback=30):
    sw_hi = float(df["High"].iloc[-lookback:].max())
    sw_lo = float(df["Low"].iloc[-lookback:].min())
    rng   = sw_hi - sw_lo
    if rng == 0:
        return sw_hi, sw_lo, {}, rng
    return sw_hi, sw_lo, {
        "236": sw_hi-rng*0.236,"382":sw_hi-rng*0.382,
        "500":sw_hi-rng*0.500, "618":sw_hi-rng*0.618,
        "786":sw_hi-rng*0.786,
        "ext127":sw_hi+rng*0.272,"ext161":sw_hi+rng*0.618,
        "ext261":sw_hi+rng*1.618,
    }, rng

def action_label(norm_score: float) -> str:
    if norm_score >= ACTION_THRESHOLDS["strong_buy"]: return "STRONG BUY"
    if norm_score >= ACTION_THRESHOLDS["buy"]:        return "BUY"
    if norm_score >= ACTION_THRESHOLDS["watch"]:      return "WATCH"
    return "SKIP"

def action_label_with_preconfirm(
    norm_score: float,
    pca_score: float = 0.0,
    em_score: float = 0.0,
    phase: str = "IDLE",
    smart_money_verdict: str = "NEUTRAL",
    accum_stage: str = "NONE",
) -> str:
    """
    GAP-5 — Extended action label with PRE-CONFIRM tier.

    PRE-CONFIRM fires when:
      • Phase is SETUP or IDLE (price hasn't confirmed yet)
      • PCAScore ≥ 55 (strong buying pressure evidence)
      • EmScore ≥ 40 (coiling mechanics present)
      • SmartMoneyVerdict in (ABSORBING, ACCUMULATING, MARKUP_READY)
      • AccumStage in (1B, 1C, 2A) — inside the base or just leaving it

    PRE-CONFIRM sits between WATCH and BUY in the action hierarchy.
    It cannot be suppressed by the breadth gate (pre-confirmation stocks
    are precisely those that haven't run yet).
    """
    base = action_label(norm_score)
    # Only consider upgrade to PRE-CONFIRM for unconfirmed phases
    if phase not in (PHASE_SETUP, PHASE_IDLE):
        return base
    # Must already pass WATCH threshold (some signal quality required)
    if norm_score < ACTION_THRESHOLDS["watch"]:
        return base
    # Gate: strong buying pressure evidence required
    if pca_score < 55:
        return base
    # Gate: coiling mechanics required (v16.0: lowered from 40 → 30 — strong PCA+SM
    # can compensate for a coil that hasn't fully tightened yet)
    if em_score < 30:
        return base
    # Gate: smart money must show at least absorption
    if smart_money_verdict not in ("ABSORBING", "ACCUMULATING", "MARKUP_READY"):
        return base
    # Gate: must be inside an identified accumulation base (not random)
    if accum_stage not in ("1B", "1C", "2A"):
        return base
    return "PRE-CONFIRM"


# ══════════════════════════════════════════════════════════════════════════════
# v16.0 — FIX-1/3/5/7: READINESS SCORE + TRADE INTENT
# Single transparent composite replacing the pile of raw sub-scores.
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# MASTER REGIME ENGINE
# Answers: "What kind of market is this right now?"
# This is the missing institutional layer ChatGPT flagged.
# VIX + breadth alone is not enough — you need market *structure* regime.
#
# Regimes and their behavioural rules:
#
#   REGIME_TREND       — broad uptrend, sector rotation active, leaders emerging.
#                        → All setups valid. Full size. PRE-CONFIRM aggressive.
#
#   REGIME_ROTATION    — trend intact but sectors churning, no clear leader.
#                        → Prefer accumulating setups. Reduce breakout size 25%.
#                          PRE-CONFIRM only on sector leaders.
#
#   REGIME_DISTRIBUTION — breadth deteriorating, A/D weakening, volume on down-days.
#                         → No new entries on breakouts. Size down 50%.
#                           Only exits and short setups.
#
#   REGIME_PANIC       — VIX spike + breadth collapse. Fear-driven selling.
#                        → No long entries. Watch for capitulation reversal.
#                          Short-sell setups only.
#
#   REGIME_EXPANSION   — new breadth high, multiple sector breakouts in sync.
#                        → Increase T3 targets 20%. Full aggression on all setups.
#
# Each regime adjusts: score thresholds, target multipliers,
# PRE-CONFIRM aggressiveness, SL width, and confidence display.
# ══════════════════════════════════════════════════════════════════════════════

REGIME_TREND        = "TREND"
REGIME_ROTATION     = "ROTATION"
REGIME_DISTRIBUTION = "DISTRIBUTION"
REGIME_PANIC        = "PANIC"
REGIME_EXPANSION    = "EXPANSION"

# Per-regime adjustment tables
_REGIME_ADJUSTMENTS = {
    REGIME_EXPANSION:    {"score_floor": 45, "target_mult": 1.20, "preconfirm": "aggressive", "sl_mult": 1.0,  "size_pct": 1.00},
    REGIME_TREND:        {"score_floor": 50, "target_mult": 1.00, "preconfirm": "normal",     "sl_mult": 1.0,  "size_pct": 1.00},
    REGIME_ROTATION:     {"score_floor": 55, "target_mult": 0.90, "preconfirm": "selective",  "sl_mult": 1.1,  "size_pct": 0.75},
    REGIME_DISTRIBUTION: {"score_floor": 65, "target_mult": 0.75, "preconfirm": "off",        "sl_mult": 1.2,  "size_pct": 0.50},
    REGIME_PANIC:        {"score_floor": 80, "target_mult": 0.60, "preconfirm": "off",        "sl_mult": 1.5,  "size_pct": 0.25},
}


def _compute_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Single-symbol ADX using Wilder's RMA — matches standard charting platform values."""
    if len(df) < period * 3:
        return 20.0
    hi  = df["High"].values.astype(np.float32)
    lo  = df["Low"].values.astype(np.float32)
    cl  = df["Close"].values.astype(np.float32)
    prev_hi = np.roll(hi, 1); prev_hi[0] = hi[0]
    prev_lo = np.roll(lo, 1); prev_lo[0] = lo[0]
    prev_cl = np.roll(cl, 1); prev_cl[0] = cl[0]
    up_m  = hi - prev_hi; dn_m = prev_lo - lo
    pdm   = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    ndm   = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
    tr    = np.maximum(hi-lo, np.maximum(np.abs(hi-prev_cl), np.abs(lo-prev_cl)))
    atr_v = _rma_np(tr, period)
    pdi   = 100*_rma_np(pdm, period)/(atr_v+1e-10)
    ndi   = 100*_rma_np(ndm, period)/(atr_v+1e-10)
    dx    = 100*np.abs(pdi-ndi)/(pdi+ndi+1e-10)
    adx_v = _rma_np(dx, period)
    return float(adx_v[-1])

def _compute_squeeze(df: pd.DataFrame, period: int = 20) -> bool:
    """Returns True if BB inside KC (squeeze on)."""
    if len(df) < period:
        return False
    cl = df["Close"].values.astype(np.float32)
    hi = df["High"].values.astype(np.float32)
    lo = df["Low"].values.astype(np.float32)
    mid   = _ema_np(cl, period)
    dev2  = _ema_np((cl-mid)**2, period)
    std   = np.sqrt(np.maximum(dev2, 0))
    bb_hi = mid + 2.0*std; bb_lo = mid - 2.0*std
    atr_k = _ema_np(np.maximum(hi-lo,
             np.maximum(np.abs(hi-np.roll(cl,1)),
                        np.abs(lo-np.roll(cl,1)))), period)
    kc_hi = mid + 1.5*atr_k; kc_lo = mid - 1.5*atr_k
    return bool(bb_hi[-1] <= kc_hi[-1] and bb_lo[-1] >= kc_lo[-1])

def _sqz_depth(df: pd.DataFrame, period: int = 20) -> float:
    """BB-width / KC-width at the last bar.
    1.0 = bands exactly equal; <1.0 = BB inside KC (squeeze); near 0 = very tight.
    UI shows  int((1 - depth) * 100) % tight, so depth=0.4 → 60% tight."""
    if len(df) < period: return 1.0
    cl = df["Close"].values.astype(np.float32)
    hi = df["High"].values.astype(np.float32)
    lo = df["Low"].values.astype(np.float32)
    mid  = _ema_np(cl, period)
    dev2 = _ema_np((cl - mid) ** 2, period)
    std  = float(np.sqrt(max(float(dev2[-1]), 0.0)))
    bb_w = 4.0 * std   # full BB width (upper - lower)
    prev = np.roll(cl, 1); prev[0] = cl[0]
    tr   = np.maximum(hi - lo, np.maximum(np.abs(hi - prev), np.abs(lo - prev)))
    atr_k = float(_ema_np(tr, period)[-1])
    kc_w  = 3.0 * atr_k   # full KC width (matches 1.5× multiplier each side)
    return round(float(np.clip(bb_w / (kc_w + 1e-10), 0.0, 2.0)), 3)

def _compute_vol_contraction(df: pd.DataFrame) -> float:
    """ATR_5 / ATR_20 ratio — <0.75 = compressed."""
    if len(df) < 25:
        return 1.0
    cl = df["Close"].values.astype(np.float32)
    hi = df["High"].values.astype(np.float32)
    lo = df["Low"].values.astype(np.float32)
    prev_cl = np.roll(cl,1); prev_cl[0] = cl[0]
    tr      = np.maximum(hi-lo, np.maximum(np.abs(hi-prev_cl), np.abs(lo-prev_cl)))
    atr5    = float(_ema_np(tr, 5)[-1])
    atr20   = float(_ema_np(tr,20)[-1])
    return atr5/(atr20+1e-10)

# ══════════════════════════════════════════════════════════════════════════════
# v15.5 — EMERGING MOMENTUM ENGINE
# Surfaces stocks BEFORE they become obvious, using 7 leading indicators.
# ══════════════════════════════════════════════════════════════════════════════

def _count_squeeze_bars(df: pd.DataFrame, period: int = 20,
                         max_lookback: int = 40) -> int:
    """Count consecutive bars currently in Keltner/BB squeeze (from most recent bar back)."""
    n = len(df)
    if n < period + 2:
        return 0
    cl = df["Close"].values.astype(np.float64)
    hi = df["High"].values.astype(np.float64)
    lo = df["Low"].values.astype(np.float64)
    count = 0
    for offset in range(min(max_lookback, n - period)):
        end = n - offset
        if end < period:
            break
        cl_w = cl[end - period:end]
        hi_w = hi[end - period:end]
        lo_w = lo[end - period:end]
        mid   = _ema_np(cl_w, period)[-1]
        dev2  = _ema_np((cl_w - _ema_np(cl_w, period)) ** 2, period)[-1]
        std   = float(np.sqrt(max(dev2, 0)))
        bb_hi = mid + 2.0 * std;  bb_lo = mid - 2.0 * std
        prev_cl_w = np.roll(cl_w, 1); prev_cl_w[0] = cl_w[0]
        tr_w   = np.maximum(hi_w - lo_w,
                 np.maximum(np.abs(hi_w - prev_cl_w), np.abs(lo_w - prev_cl_w)))
        atr_k  = float(_ema_np(tr_w, period)[-1])
        kc_hi  = mid + 1.5 * atr_k;  kc_lo = mid - 1.5 * atr_k
        if bb_hi <= kc_hi and bb_lo >= kc_lo:
            count += 1
        else:
            break
    return count


def compute_emerging_score(
    df: pd.DataFrame,
    mode: str,
    nifty_close: pd.Series,
    rs_rank: int = 50,
) -> dict:
    """
    Emerging Momentum Score (0–100) — surfaces stocks BEFORE they become obvious.

    Components (max pts):
    1. RS Acceleration      15 — relative strength gaining speed vs index
    2. ATR Compression      15 — volatility coiling toward a breakout
    3. RVOL Acceleration    15 — volume building quietly (smart-money fingerprint)
    4. EMA Convergence      15 — fast/slow EMAs tightening = decision approaching
    5. Squeeze Pressure     15 — consecutive BB-inside-KC bars = stored energy
    6. Sector Momentum      10 — sector tailwind (enriched post-scan in run_scan)
    7. Opening Range Exp.   15 — price expanding beyond recent consolidation

    Labels: IGNITING ≥65 · BUILDING ≥50 · COILING ≥35 · LATENT ≥20 · QUIET <20
    """
    out = dict(
        EmScore=0.0, EmLabel="QUIET",
        EmRSAccel=0.0, EmATRCompress=0.0, EmRVolAccel=0.0,
        EmEMAConv=0.0, EmSqzPressure=0.0, EmSectorMom=0.0, EmORExpansion=0.0,
    )
    try:
        if df is None or len(df) < 40:
            return out
        cl  = df["Close"].values.astype(np.float64)
        hi  = df["High"].values.astype(np.float64)
        lo  = df["Low"].values.astype(np.float64)
        vol = df["Volume"].values.astype(np.float64)
        n   = len(cl)
        cfg = MODE_CFG[mode]
        ef_span = cfg["ema_fast"]
        es_span = cfg["ema_slow"]

        # ── 1. RS ACCELERATION (0–15 pts) ────────────────────────────────────
        # RS is accelerating when recent outperformance > medium > long window
        rs_pts = 0.0
        try:
            nifty = (nifty_close.values.astype(np.float64)
                     if nifty_close is not None and len(nifty_close) >= 20
                     else None)
            if nifty is not None:
                def _rs(bars):
                    if n < bars + 1 or len(nifty) < bars + 1:
                        return 0.0
                    s = (cl[-1] - cl[-bars]) / (cl[-bars] + 1e-10) * 100
                    m = (nifty[-1] - nifty[-bars]) / (nifty[-bars] + 1e-10) * 100
                    return s - m
                rs5, rs10, rs20 = _rs(5), _rs(10), _rs(20)
                if rs5 > rs10 > 0:          # Accelerating outperformance
                    rs_pts = min(15.0, (rs5 - rs10) * 2.5 + 5)
                elif rs5 > 0 and rs5 > rs20 * 0.5:
                    rs_pts = min(8.0, rs5 * 0.6)
                if rs_rank >= 70 and rs5 > 0:  # High rank + still accelerating
                    rs_pts = min(15.0, rs_pts + 3)
        except Exception:
            pass

        # ── 2. ATR COMPRESSION (0–15 pts) ────────────────────────────────────
        # Volatility contracting → coiling energy before expansion
        atr_pts = 0.0
        try:
            if n >= 25:
                prev_cl = np.roll(cl, 1); prev_cl[0] = cl[0]
                tr_arr  = np.maximum(hi - lo, np.maximum(
                          np.abs(hi - prev_cl), np.abs(lo - prev_cl)))
                atr5_now  = float(_ema_np(tr_arr, 5)[-1])
                atr20_now = float(_ema_np(tr_arr, 20)[-1])
                ratio_now = atr5_now / (atr20_now + 1e-10)
                if   ratio_now < 0.65: atr_pts = 15.0
                elif ratio_now < 0.75: atr_pts = 12.0
                elif ratio_now < 0.85: atr_pts = 8.0
                elif ratio_now < 0.95: atr_pts = 4.0
                # Bonus: actively compressing (trend in ratio)
                if n > 15:
                    atr5_5  = float(_ema_np(tr_arr[:-5], 5)[-1])
                    atr20_5 = float(_ema_np(tr_arr[:-5], 20)[-1])
                    if ratio_now < atr5_5 / (atr20_5 + 1e-10) - 0.05:
                        atr_pts = min(15.0, atr_pts + 4.0)
        except Exception:
            pass

        # ── 3. RVOL ACCELERATION (0–15 pts) ──────────────────────────────────
        # Volume building quietly across successive windows → smart money
        rvol_pts = 0.0
        try:
            if n >= 20:
                avg_vol = float(np.mean(vol[-21:-1])) if n >= 22 else float(np.mean(vol[:-1]))
                if avg_vol > 0:
                    v_now   = float(np.mean(vol[-3:]))
                    v_5ago  = float(np.mean(vol[-8:-5]))   if n >= 8  else avg_vol
                    v_10ago = float(np.mean(vol[-13:-10])) if n >= 13 else avg_vol
                    r_now   = v_now   / avg_vol
                    r_5ago  = v_5ago  / avg_vol
                    r_10ago = v_10ago / avg_vol
                    if r_now > r_5ago > r_10ago and r_now > 0.8:  # Sequential build
                        rvol_pts = min(15.0, (r_now - r_10ago) * 15)
                    elif r_now > r_5ago and r_now > 0.9:
                        rvol_pts = min(9.0, (r_now - r_5ago) * 12)
                    elif 0.4 < r_now < 0.75:                       # Quiet dryup = stealth accumulation
                        rvol_pts = 5.0
        except Exception:
            pass

        # ── 4. EMA CONVERGENCE (0–15 pts) ────────────────────────────────────
        # Fast + slow EMAs tightening → coiling, decision point approaching
        conv_pts = 0.0
        try:
            if n >= es_span + 5:
                ef_arr = _ema_np(cl, ef_span)
                es_arr = _ema_np(cl, es_span)
                dist_now   = abs(ef_arr[-1]  - es_arr[-1])
                dist_5ago  = abs(ef_arr[-6]  - es_arr[-6])  if n >= 6  else dist_now
                dist_10ago = abs(ef_arr[-11] - es_arr[-11]) if n >= 11 else dist_now
                c_last     = cl[-1] + 1e-10
                dpct_now   = dist_now   / c_last * 100
                dpct_5ago  = dist_5ago  / c_last * 100
                dpct_10ago = dist_10ago / c_last * 100
                converging = dpct_now < dpct_5ago
                if   converging and dpct_now < 0.5:  conv_pts = 15.0
                elif converging and dpct_now < 1.0:  conv_pts = 11.0
                elif converging and dpct_now < 2.0:  conv_pts = 7.0
                elif converging:                     conv_pts = 4.0
                elif dpct_now < dpct_10ago * 0.65:   conv_pts = 6.0  # 35% tighter in 10 bars
                # Bonus: bullish convergence (fast still > slow)
                if ef_arr[-1] > es_arr[-1] and converging:
                    conv_pts = min(15.0, conv_pts + 4.0)
        except Exception:
            pass

        # ── 5. SQUEEZE PRESSURE (0–15 pts) ───────────────────────────────────
        # Consecutive bars in BB/KC squeeze → more bars = more stored kinetic energy
        sqz_pts = 0.0
        try:
            csq = _count_squeeze_bars(df, period=20, max_lookback=40)
            if   csq >= 20: sqz_pts = 15.0
            elif csq >= 15: sqz_pts = 12.0
            elif csq >= 10: sqz_pts = 9.0
            elif csq >= 5:  sqz_pts = 6.0
            elif csq >= 2:  sqz_pts = 3.0
        except Exception:
            pass

        # ── 6. SECTOR MOMENTUM (0–10 pts) — placeholder enriched in run_scan ─
        sec_pts = 0.0

        # ── 7. OPENING RANGE EXPANSION (0–15 pts) ────────────────────────────
        # Price moving beyond recent consolidation zone = first activation signal
        or_pts = 0.0
        try:
            if mode == "Intraday" and n >= 8:
                # First 6 bars (~30 min at 5m) = opening range
                or_hi = float(np.max(hi[:6]))
                or_lo = float(np.min(lo[:6]))
                or_rng = or_hi - or_lo
                cur = cl[-1]
                if or_rng > 0:
                    exp_pct = (cur - or_hi) / or_rng
                    if   exp_pct > 0.20:  or_pts = 15.0
                    elif exp_pct > 0.05:  or_pts = 10.0
                    elif exp_pct >= 0:    or_pts = 6.0
                    elif exp_pct > -0.15: or_pts = 3.0
            else:
                lb = min(10 if mode == "Swing" else 20, n - 2)
                rng_hi = float(np.max(hi[-lb - 1:-1]))
                rng_lo = float(np.min(lo[-lb - 1:-1]))
                rng_w  = rng_hi - rng_lo
                cur    = cl[-1]
                if rng_w > 0:
                    exp_pct = (cur - rng_hi) / rng_w
                    if   exp_pct > 0.15:  or_pts = 15.0
                    elif exp_pct > 0.02:  or_pts = 10.0
                    elif exp_pct >= 0:    or_pts = 6.0
                    elif cur > rng_lo + rng_w * 0.65: or_pts = 3.0
        except Exception:
            pass

        # ── TOTAL ─────────────────────────────────────────────────────────────
        total = round(float(np.clip(
            rs_pts + atr_pts + rvol_pts + conv_pts + sqz_pts + sec_pts + or_pts,
            0, 100)), 1)
        label = ("IGNITING" if total >= 65 else "BUILDING" if total >= 50
                 else "COILING" if total >= 35 else "LATENT" if total >= 20 else "QUIET")
        out.update(
            EmScore       = total,
            EmLabel       = label,
            EmRSAccel     = round(rs_pts,   1),
            EmATRCompress = round(atr_pts,  1),
            EmRVolAccel   = round(rvol_pts, 1),
            EmEMAConv     = round(conv_pts, 1),
            EmSqzPressure = round(sqz_pts,  1),
            EmSectorMom   = 0.0,   # filled by enrich_sector_momentum() in run_scan
            EmORExpansion = round(or_pts,   1),
        )
    except Exception:
        pass
    return out

# ══════════════════════════════════════════════════════════════════════════════
# v15.6 — PRE-CONFIRMATION ACCUMULATION (PCA) ENGINE
# Detects institutional buying BEFORE price confirms — 7 signal components.
# This layer completes the early-accumulation intelligence stack that
# EmScore starts but doesn't finish (EmScore targets coiling mechanics;
# PCA targets the buying-pressure fingerprint beneath the coil).
# ══════════════════════════════════════════════════════════════════════════════

_PCA_COMPONENTS = [
    ("Rel CMF",       "PCACMFRel",       15, "💧"),
    ("Vol Cmp Seq",   "PCAVolCmpSeq",    15, "🗜"),
    ("Hidden Accum",  "PCAHiddenAccum",  15, "👻"),
    ("Effort/Result", "PCAEffortResult", 15, "⚖"),
    ("Range Persist", "PCARangeCont",    10, "📏"),
    ("Failed BRK",    "PCAFailedBrkdn",  15, "🛡"),
    ("Vol Asymmetry", "PCAVolAsym",      15, "⚖"),
]

