"""
patterns.py — Price pattern detection: VCP, Fib, Darvas, harmonic, candle, MTF.
"""
import concurrent.futures
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional

from config import MODE_CFG
from indicators import ema, rsi, atr_series
from data_fetch import fetch_async, _CACHE_DIR

def _tol(atr_val: float, mode: str) -> float:
    """Absolute price tolerance for pivot comparisons."""
    return atr_val * _PIVOT_TOL.get(mode, 0.15)

def _closed_candle_df(df: pd.DataFrame, mode: str, market_open: bool) -> pd.DataFrame:
    """Strip the live forming bar for Intraday when market is open."""
    if mode == "Intraday" and market_open and len(df) > 1:
        return df.iloc[:-1]
    return df

def detect_vcp(df: pd.DataFrame, atr_val: float = 0.0, mode: str = "Swing",
               min_contractions: int = 2, lookback: int = 60) -> dict:
    result = dict(detected=False, n_contractions=0, tightest_pct=0.0, vcp_grade="NONE")
    if len(df) < max(lookback, 20): return result
    sl = df.iloc[-lookback:]
    hi = sl["High"].values.astype(np.float64); lo = sl["Low"].values.astype(np.float64)
    vol = sl["Volume"].values.astype(np.float64); n = len(sl)
    tol_price = _tol(atr_val, mode) if atr_val > 0 else 0.0
    wing = 3
    phi = [i for i in range(wing, n-wing) if hi[i] >= np.max(hi[i-wing:i+wing+1]) - tol_price]
    plo = [i for i in range(wing, n-wing) if lo[i] <= np.min(lo[i-wing:i+wing+1]) + tol_price]
    if len(phi) < 2 or len(plo) < 2: return result
    segments = []
    for ph in phi:
        sub = [pl for pl in plo if pl > ph]
        if not sub: continue
        pl = sub[0]
        depth_abs = hi[ph] - lo[pl]; depth_pct = depth_abs / hi[ph] * 100
        seg_vol = float(np.mean(vol[ph:pl+1])) if pl > ph else float(vol[ph])
        segments.append((ph, pl, depth_pct, seg_vol, depth_abs))
    if len(segments) < 2: return result
    n_cont = 0
    for i in range(len(segments)-1, 0, -1):
        cur = segments[i]; prev = segments[i-1]
        if (cur[2] < prev[2]*0.95 and cur[3] < prev[3]*0.95
                and (prev[4]-cur[4]) > tol_price):
            n_cont += 1
        else: break
    detected = n_cont >= min_contractions
    tightest_pct = float(segments[-1][2]) if segments else 0.0
    grade = "PERFECT" if n_cont>=4 else "GOOD" if n_cont>=3 else "FORMING" if n_cont>=2 else "NONE"
    result.update(detected=detected, n_contractions=n_cont,
                  tightest_pct=round(tightest_pct,2), vcp_grade=grade)
    return result

def compute_anchored_vwap(df: pd.DataFrame, atr_val: float = 0.0,
                           mode: str = "Swing", lookback: int = 60) -> dict:
    result = dict(avwap=None, anchor_idx=None, pct_above=0.0,
                  price_above=False, near_support=False)
    if not {"High","Low","Close","Volume"}.issubset(df.columns) or len(df) < 10:
        return result
    sl = df.iloc[-lookback:]
    closes = sl["Close"].values.astype(np.float64); highs = sl["High"].values.astype(np.float64)
    lows = sl["Low"].values.astype(np.float64); volumes = sl["Volume"].values.astype(np.float64)
    n = len(sl); avg_vol = float(np.mean(volumes)) or 1.0
    best_idx = None; best_cl = float("inf")
    for i in range(n-1):
        if closes[i] < best_cl and volumes[i] >= avg_vol*0.8:
            best_cl = closes[i]; best_idx = i
    if best_idx is None: best_idx = int(np.argmin(closes[:-1]))
    typical = (highs[best_idx:]+lows[best_idx:]+closes[best_idx:])/3.0
    vols_s = volumes[best_idx:]
    avwap = float(np.cumsum(typical*vols_s)[-1] / (np.cumsum(vols_s)[-1]+1e-10))
    current = float(closes[-1])
    tol_abs = _tol(atr_val, mode) if atr_val > 0 else avwap*0.01
    pct_above = (current-avwap)/avwap*100 if avwap > 0 else 0.0
    price_above = current > avwap-tol_abs
    near_support = price_above and (current-avwap) < tol_abs
    result.update(avwap=round(avwap,2), anchor_idx=n-1-best_idx,
                  pct_above=round(pct_above,2), price_above=price_above,
                  near_support=near_support)
    return result

def score_fib_pullback(df: pd.DataFrame, atr_val: float,
                        mode: str = "Swing", lookback: int = 60) -> dict:
    result = dict(quality=0, grade="POOR", depth_ok=False, vol_ok=False,
                  recovery_ok=False, fib_level="—")
    if len(df) < 20 or atr_val <= 0: return result
    sl = df.iloc[-lookback:]
    hi_a = sl["High"].values.astype(np.float64); lo_a = sl["Low"].values.astype(np.float64)
    cl_a = sl["Close"].values.astype(np.float64); vo_a = sl["Volume"].values.astype(np.float64)
    n = len(sl); tol_abs = _tol(atr_val, mode); wing = 3
    phi = [i for i in range(wing, n-wing) if hi_a[i] >= np.max(hi_a[i-wing:i+wing+1])-tol_abs]
    plo = [i for i in range(wing, n-wing) if lo_a[i] <= np.min(lo_a[i-wing:i+wing+1])+tol_abs]
    if not phi or not plo: return result
    sw_hi_i = phi[-1]
    prior_lo = [i for i in plo if i < sw_hi_i]
    if not prior_lo: return result
    sw_lo_i = prior_lo[-1]
    sw_hi = float(hi_a[sw_hi_i]); sw_lo = float(lo_a[sw_lo_i]); rng = sw_hi-sw_lo
    if rng < atr_val*0.5: return result
    post_lo = float(np.min(lo_a[sw_hi_i:])); post_cl = float(cl_a[-1])
    depth_pct = (sw_hi-post_lo)/rng*100; tol_pct = tol_abs/rng*100
    def _in_zone(lo_p, hi_p): return (lo_p-tol_pct) <= depth_pct <= (hi_p+tol_pct)
    if _in_zone(38.2,50.0):   ds=40; dok=True; fl="38.2–50"
    elif _in_zone(50.0,61.8): ds=30; dok=True; fl="50–61.8"
    elif _in_zone(23.6,38.2): ds=15; dok=False; fl="23.6–38.2"
    elif _in_zone(61.8,78.6): ds=10; dok=False; fl="61.8–78.6"
    else:                     ds=0;  dok=False; fl="Outside"
    adv_v=float(np.mean(vo_a[sw_lo_i:sw_hi_i+1])) if sw_hi_i>sw_lo_i else 1.0
    pb_v=float(np.mean(vo_a[sw_hi_i:])) if len(vo_a[sw_hi_i:])>0 else 1.0
    vr=pb_v/(adv_v+1e-10)
    vs=30 if vr<=0.60 else (20 if vr<=0.75 else (10 if vr<=0.90 else 0))
    vok=vr<=0.75
    f500=sw_hi-rng*0.500; f618=sw_hi-rng*0.618; overshoot=max(0.0,f500-post_lo)
    if post_cl>f500-tol_abs and overshoot<=tol_abs*2: rs=30; rok=True
    elif post_cl>f618-tol_abs: rs=15; rok=False
    else: rs=0; rok=False
    quality=ds+vs+rs
    grade="EXCELLENT" if quality>=80 else "GOOD" if quality>=60 else "FAIR" if quality>=40 else "POOR"
    result.update(quality=quality, grade=grade, depth_ok=dok, vol_ok=vok,
                  recovery_ok=rok, fib_level=fl)
    return result

def detect_volume_dryup(df: pd.DataFrame, atr_val: float,
                         mode: str = "Swing", window: int = 5) -> dict:
    result = dict(dry_up=False, intensity=0, bars=0, vol_pct=100.0)
    if len(df) < max(window+5, 25) or atr_val <= 0: return result
    vols = df["Volume"].values.astype(np.float64)
    highs = df["High"].values.astype(np.float64); lows = df["Low"].values.astype(np.float64)
    n = len(vols)
    avg_vol_20 = float(np.mean(vols[-21:-1])) if n>=22 else float(np.mean(vols[:-1]))
    if avg_vol_20 <= 0: return result
    consec = 0
    for i in range(1, min(window+1, n)):
        if vols[-i] < vols[-(i+1)]: consec += 1
        else: break
    tight = False
    if consec >= 2:
        tight = (float(np.max(highs[-consec:]))-float(np.min(lows[-consec:]))) < atr_val*1.2
    latest_vol_pct = float(vols[-1])/avg_vol_20*100
    dry_up = consec>=2 and tight and latest_vol_pct<80.0
    if dry_up:
        intensity = 3 if (latest_vol_pct<40 and consec>=4) else (2 if (latest_vol_pct<60 and consec>=3) else 1)
    else: intensity = 0
    result.update(dry_up=dry_up, intensity=intensity, bars=consec, vol_pct=round(latest_vol_pct,1))
    return result

def compute_relative_volume(df: pd.DataFrame, lookback: int = 60) -> dict:
    result = dict(rel_vol_pct=50.0, label="NORMAL", ratio=1.0)
    if len(df) < 10: return result
    vols = df["Volume"].values.astype(np.float64)
    window = vols[-lookback-1:-1] if len(vols)>lookback+1 else vols[:-1]
    cur_vol = float(vols[-1])
    if len(window)==0 or float(np.max(window))==0: return result
    pct_rank = float(np.sum(window<cur_vol))/len(window)*100
    ratio = cur_vol/(float(np.mean(window))+1e-10)
    label = "SURGE" if pct_rank>=85 else "HIGH" if pct_rank>=65 else "NORMAL" if pct_rank>=30 else "DRY"
    result.update(rel_vol_pct=round(pct_rank,1), label=label, ratio=round(ratio,2))
    return result

def detect_darvas_box(df: pd.DataFrame, atr_val: float,
                       mode: str = "Swing", lookback: int = 60) -> dict:
    result = dict(in_box=False, breakout=False, box_top=0.0, box_bottom=0.0,
                  box_width_pct=0.0, bars_in_box=0)
    if len(df) < 20 or atr_val <= 0: return result
    sl = df.iloc[-lookback:]
    hi_a = sl["High"].values.astype(np.float64); lo_a = sl["Low"].values.astype(np.float64)
    cl_a = sl["Close"].values.astype(np.float64); n = len(sl)
    tol = _tol(atr_val, mode)
    peak_i = int(np.argmax(hi_a)); box_top = float(hi_a[peak_i])
    if peak_i >= n-3: peak_i = max(0, peak_i-3)
    top_confirmed = False; top_i = peak_i; consec_below = 0
    for i in range(peak_i+1, min(peak_i+10, n)):
        if hi_a[i] < box_top+tol: consec_below += 1
        else: box_top = float(hi_a[i]); consec_below = 0
        if consec_below >= 3: top_confirmed = True; top_i = i; break
    if not top_confirmed: return result
    sub_lo = lo_a[top_i:]
    if len(sub_lo) < 4: return result
    trough_i = int(np.argmin(sub_lo))+top_i; box_bottom = float(lo_a[trough_i])
    btm_confirmed = False; consec_above = 0
    for i in range(trough_i+1, min(trough_i+10, n)):
        if lo_a[i] > box_bottom-tol: consec_above += 1
        else: box_bottom = float(lo_a[i]); consec_above = 0
        if consec_above >= 3: btm_confirmed = True; break
    if not btm_confirmed: return result
    cur = float(cl_a[-1])
    in_box = (box_bottom-tol) <= cur <= (box_top+tol)
    breakout = cur > box_top+tol
    box_width_pct = (box_top-box_bottom)/box_bottom*100 if box_bottom>0 else 0.0
    result.update(in_box=in_box, breakout=breakout, box_top=round(box_top,2),
                  box_bottom=round(box_bottom,2), box_width_pct=round(box_width_pct,2),
                  bars_in_box=n-trough_i)
    return result

def score_all_patterns(df: pd.DataFrame, atr_val: float,
                        mode: str = "Swing", market_open: bool = False) -> tuple:
    """FIX-A+B: runs on closed candles only; called externally via enrich_with_patterns."""
    closed = _closed_candle_df(df, mode, market_open)
    if len(closed) < 20: return 0, {}
    vcp    = detect_vcp(closed,            atr_val=atr_val, mode=mode)
    avwap  = compute_anchored_vwap(closed, atr_val=atr_val, mode=mode)
    fibq   = score_fib_pullback(closed,    atr_val=atr_val, mode=mode)
    vdu    = detect_volume_dryup(closed,   atr_val=atr_val, mode=mode)
    rvol   = compute_relative_volume(closed)
    darvas = detect_darvas_box(closed,     atr_val=atr_val, mode=mode)
    pts = 0
    if vcp["n_contractions"] >= 3:    pts += 14
    elif vcp["n_contractions"] >= 2:  pts += 7
    if avwap["price_above"]:          pts += 8
    if avwap["near_support"]:         pts += 4
    fq = fibq["quality"]
    if fq >= 75:                      pts += 10
    elif fq >= 50:                    pts += 5
    if vdu["intensity"] >= 2:         pts += 8
    elif vdu["intensity"] == 1:       pts += 4
    rvp = rvol["rel_vol_pct"]
    if rvp >= 85:                     pts += 10
    elif rvp >= 65:                   pts += 5
    if darvas["breakout"]:            pts += 12
    elif darvas["in_box"]:            pts += 6
    patterns = dict(vcp=vcp, avwap=avwap, fib_quality=fibq,
                    vol_dryup=vdu, rel_vol=rvol, darvas=darvas,
                    total_pattern_pts=pts)
    return pts, patterns

# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 1 — MULTI-TIMEFRAME MOMENTUM SYNCHRONIZATION
# ══════════════════════════════════════════════════════════════════════════════

_MTF_CFG = {
    "Intraday":   (("5m",  "15m", "1h"),  (0.25, 0.40, 0.35)),
    "Swing":      (("1d",  "1wk", "1mo"), (0.30, 0.40, 0.30)),
    "Positional": (("1d",  "1wk", "1mo"), (0.20, 0.40, 0.40)),
}

def _mtf_tf_score(close_s: pd.Series, ema_fast: int, ema_slow: int) -> float:
    """Score a single timeframe: -1.0 (full bear) to +1.0 (full bull)."""
    n = len(close_s)
    if n < ema_slow + 5:
        return 0.0
    c   = float(close_s.iloc[-1])
    ef  = float(close_s.ewm(span=ema_fast, adjust=False).mean().iloc[-1])
    es  = float(close_s.ewm(span=ema_slow, adjust=False).mean().iloc[-1])
    rv  = float(rsi(close_s, 14).iloc[-1])
    lb  = max(1, min(21, n - 1))
    mom = (c - float(close_s.iloc[-lb])) / float(close_s.iloc[-lb]) * 100
    s   = 0.0
    s  += 0.30 if c  > ef  else -0.30
    s  += 0.30 if ef > es  else -0.30
    s  += 0.20 if rv > 50  else -0.20
    s  += 0.20 if mom > 0  else -0.20
    return float(np.clip(s, -1.0, 1.0))

def compute_mtf_sync(sym: str, mode: str,
                     prefetched: dict | None = None) -> dict:
    """
    Multi-Timeframe Momentum Synchronization.
    Returns sync_score (0–100), alignment flag, per-TF scores, divergence flag.
    """
    out = dict(sync_score=50.0, aligned=False, bull_count=0, bear_count=0,
               tf_scores={}, divergence=False, mtf_label="NEUTRAL")
    try:
        intervals, weights = _MTF_CFG[mode]
        cfg      = MODE_CFG[mode]
        ef_span  = cfg["ema_fast"]
        es_span  = cfg["ema_slow"]
        data     = prefetched or {}
        tf_scores: dict = {}

        for tf in intervals:
            df = data.get(tf)
            if df is None or df.empty or len(df) < 30:
                tf_scores[tf] = 0.0
                continue
            tf_scores[tf] = _mtf_tf_score(df["Close"], ef_span, es_span)

        weighted  = sum(tf_scores.get(tf, 0.0) * w
                        for tf, w in zip(intervals, weights))
        sync_score = round((weighted + 1.0) / 2.0 * 100.0, 1)

        scores    = [tf_scores.get(tf, 0.0) for tf in intervals]
        bull_cnt  = sum(1 for s in scores if s >  0.2)
        bear_cnt  = sum(1 for s in scores if s < -0.2)
        aligned   = (bull_cnt == len(intervals)) or (bear_cnt == len(intervals))
        diverge   = (len(scores) >= 2
                     and ((scores[0] > 0.3 and scores[-1] < -0.3)
                          or (scores[0] < -0.3 and scores[-1] > 0.3)))

        if   sync_score >= 70 and aligned: lbl = "BULL SYNC"
        elif sync_score >= 60:             lbl = "BULL LEAN"
        elif sync_score <= 30 and aligned: lbl = "BEAR SYNC"
        elif sync_score <= 40:             lbl = "BEAR LEAN"
        elif diverge:                      lbl = "DIVERGE"
        else:                              lbl = "NEUTRAL"

        out.update(sync_score=sync_score, aligned=aligned,
                   bull_count=bull_cnt, bear_count=bear_cnt,
                   tf_scores=tf_scores, divergence=diverge, mtf_label=lbl)
    except Exception:
        pass
    return out

def prefetch_mtf_parallel(symbols: list, mode: str) -> dict:
    """Batch-fetch secondary/tertiary TF data for all survivors (async)."""
    intervals, _ = _MTF_CFG[mode]
    result: dict = {sym: {} for sym in symbols}
    if len(intervals) < 2 or not symbols:
        return result
    for tf in intervals[1:]:
        period = "1y" if tf in ("15m", "1h") else "3y"
        raw    = fetch_async(symbols, period, tf, concurrency=20)
        for sym, df in raw.items():
            result[sym][tf] = df
    return result

# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 2 — INSTITUTIONAL VOLUME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_institutional_volume(df: pd.DataFrame,
                                  mode: str = "Swing") -> dict:
    """
    Detect institutional accumulation / distribution via OBV, CMF,
    Acc/Dist line, block-volume fingerprint, and Wyckoff Effort-vs-Result.
    Returns inst_score (0–100), verdict, and component values.
    """
    out = dict(inst_score=50.0, verdict="NEUTRAL", obv_trend=0.0,
               cmf=0.0, acc_dist=0.0, block_days=0,
               effort_vs_result="NEUTRAL", inst_label="INST~")
    try:
        if len(df) < 30:
            return out
        cl  = df["Close"].values.astype(np.float64)
        hi  = df["High"].values.astype(np.float64)
        lo  = df["Low"].values.astype(np.float64)
        vol = df["Volume"].values.astype(np.float64)
        n   = len(cl)

        # OBV trend — EMA(10) vs EMA(30)
        direction  = np.sign(np.diff(cl, prepend=cl[0]))
        obv        = np.cumsum(direction * vol)
        obv_trend  = 1.0 if float(_ema_np(obv, 10)[-1]) > float(_ema_np(obv, 30)[-1]) else -1.0

        # Chaikin Money Flow (20-bar)
        win   = min(20, n)
        hlr   = np.where((hi[-win:] - lo[-win:]) == 0, 1e-10,
                          hi[-win:] - lo[-win:])
        mfm   = ((cl[-win:] - lo[-win:]) - (hi[-win:] - cl[-win:])) / hlr
        cmf   = float(np.sum(mfm * vol[-win:]) / (np.sum(vol[-win:]) + 1e-10))

        # Accumulation / Distribution
        hlr_full = np.where((hi - lo) == 0, 1e-10, hi - lo)
        ad_mfm   = ((cl - lo) - (hi - cl)) / hlr_full
        ad_line  = np.cumsum(ad_mfm * vol)
        ad_trend = 1.0 if ad_line[-1] > ad_line[-min(10, n)] else -1.0

        # Block volume (institutional fingerprint) — days > 2.5× avg
        avg_vol    = float(np.mean(vol[-min(60, n):])) or 1.0
        block_days = int(np.sum(vol[-min(20, n):] > avg_vol * 2.5))

        # Wyckoff Effort-vs-Result (last 5 bars)
        recent     = min(5, n)
        avg_rng    = float(np.mean(hi[-min(20,n):] - lo[-min(20,n):])) or 1e-10
        last_rng   = float(np.mean(hi[-recent:] - lo[-recent:]))
        last_vr    = float(np.mean(vol[-recent:])) / avg_vol
        if   last_vr > 1.3 and last_rng > avg_rng * 0.8: evr = "THRUST"
        elif last_vr > 1.3 and last_rng < avg_rng * 0.6: evr = "ABSORPTION"
        elif last_vr < 0.7:                               evr = "DRY"
        else:                                             evr = "NEUTRAL"

        # Composite score (centre 50)
        score  = 50.0
        score += obv_trend * 12.0
        score += float(np.clip(cmf * 100, -15, 15))
        score += ad_trend  * 8.0
        score += min(block_days * 3, 12)
        if evr == "THRUST":        score += 10.0
        elif evr == "ABSORPTION":  score -=  8.0
        elif evr == "DRY":         score -=  5.0
        score = float(np.clip(score, 0, 100))

        if   score >= 70: verdict = "ACCUMULATION"
        elif score >= 58: verdict = "MILD ACCUM"
        elif score <= 30: verdict = "DISTRIBUTION"
        elif score <= 42: verdict = "MILD DIST"
        else:             verdict = "NEUTRAL"

        inst_label = "INST↑" if score >= 65 else ("INST↓" if score <= 35 else "INST~")

        out.update(inst_score=round(score, 1), verdict=verdict,
                   obv_trend=round(obv_trend, 2), cmf=round(cmf, 4),
                   acc_dist=round(ad_trend, 2), block_days=block_days,
                   effort_vs_result=evr, inst_label=inst_label)
    except Exception:
        pass
    return out

# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 3 — HARMONIC / ABCD PATTERN ENGINE
# ══════════════════════════════════════════════════════════════════════════════

_HARMONIC_DEF = {
    "Gartley":   dict(AB_XA=(0.618,0.618), BC_AB=(0.382,0.886),
                      CD_BC=(1.272,1.618), AD_XA=(0.786,0.786), tol=0.05),
    "Bat":       dict(AB_XA=(0.382,0.500), BC_AB=(0.382,0.886),
                      CD_BC=(1.618,2.618), AD_XA=(0.886,0.886), tol=0.05),
    "Butterfly": dict(AB_XA=(0.786,0.786), BC_AB=(0.382,0.886),
                      CD_BC=(1.618,2.618), AD_XA=(1.272,1.272), tol=0.06),
    "Crab":      dict(AB_XA=(0.382,0.618), BC_AB=(0.382,0.886),
                      CD_BC=(2.618,3.618), AD_XA=(1.618,1.618), tol=0.06),
    "Cypher":    dict(AB_XA=(0.382,0.618), BC_AB=(1.272,1.414),
                      CD_BC=(0.382,0.786), AD_XA=(0.786,0.786), tol=0.07),
}

def _fib_ok(val: float, lo: float, hi: float, tol: float) -> bool:
    mn, mx = min(lo, hi), max(lo, hi)
    return mn * (1 - tol) <= val <= mx * (1 + tol)

def _find_swing_pivots(arr: np.ndarray, wing: int = 4) -> list:
    """Return alternating (idx, price, 'H'/'L') swing pivots."""
    n = len(arr); pivots = []
    for i in range(wing, n - wing):
        w = arr[i - wing: i + wing + 1]
        if arr[i] == np.max(w):   pivots.append((i, float(arr[i]), "H"))
        elif arr[i] == np.min(w): pivots.append((i, float(arr[i]), "L"))
    # Keep only alternating, prefer stronger pivot on same type run
    deduped: list = []
    for p in pivots:
        if deduped and deduped[-1][2] == p[2]:
            if (p[2] == "H" and p[1] > deduped[-1][1]) or \
               (p[2] == "L" and p[1] < deduped[-1][1]):
                deduped[-1] = p
        else:
            deduped.append(p)
    return deduped

def detect_harmonic_patterns(df: pd.DataFrame,
                              mode: str = "Swing") -> dict:
    """
    Detect ABCD and named harmonic patterns (Gartley, Bat, Butterfly, Crab, Cypher).
    Returns best match: pattern name, direction, quality (0–100),
    completion zone, and harmonic_score contribution.
    """
    out = dict(pattern=None, direction=None, quality=0,
               completion_zone=(0.0, 0.0), harmonic_score=0,
               d_level=0.0, detected=False)
    try:
        if len(df) < 60:
            return out
        hi  = df["High"].values.astype(np.float64)
        lo  = df["Low"].values.astype(np.float64)
        mid = (hi + lo) / 2.0

        pivots = _find_swing_pivots(mid, wing=4)
        if len(pivots) < 5:
            return out

        best: dict | None = None
        best_q = 0

        for start in range(max(0, len(pivots) - 5), -1, -1):
            pts = pivots[start: start + 5]
            if len(pts) < 5:
                continue
            X, A, B, C, D = pts
            types = [p[2] for p in pts]
            # Strict alternation required
            if any(types[i] == types[i+1] for i in range(4)):
                continue

            px, pa, pb, pc, pd_ = [p[1] for p in pts]
            bull = types[0] == "L"   # bullish: X=low, completion at D=low

            XA  = abs(pa - px)
            AB  = abs(pa - pb)
            BC  = abs(pc - pb)
            CD  = abs(pc - pd_)
            if any(v <= 0 for v in (XA, AB, BC, CD)):
                continue

            ab_xa = AB / XA
            bc_ab = BC / AB
            cd_bc = CD / BC
            ad_xa = abs(pa - pd_) / XA

            # ABCD (simple)
            abcd_q = 0
            if _fib_ok(ab_xa, 0.382, 0.786, 0.07) and \
               _fib_ok(cd_bc, 1.13,  1.618, 0.08):
                abcd_q = 60

            # Named harmonics
            for name, r in _HARMONIC_DEF.items():
                tol = r["tol"]
                hits = sum([_fib_ok(ab_xa, *r["AB_XA"], tol),
                            _fib_ok(bc_ab, *r["BC_AB"], tol),
                            _fib_ok(cd_bc, *r["CD_BC"], tol),
                            _fib_ok(ad_xa, *r["AD_XA"], tol)])
                q = int(hits / 4 * 100)
                if hits >= 3 and q > best_q:
                    d_lo = (pc - XA * r["AD_XA"][0] if bull
                            else pc + XA * r["AD_XA"][0])
                    d_hi = (pc - XA * r["AD_XA"][1] if bull
                            else pc + XA * r["AD_XA"][1])
                    best_q = q
                    best   = dict(pattern=name,
                                  direction="BULL" if bull else "BEAR",
                                  quality=q,
                                  completion_zone=(round(min(d_lo,d_hi),2),
                                                   round(max(d_lo,d_hi),2)),
                                  d_level=round(pd_, 2), detected=True)

            if abcd_q > best_q and best is None:
                best_q = abcd_q
                best   = dict(pattern="ABCD",
                              direction="BULL" if bull else "BEAR",
                              quality=abcd_q,
                              completion_zone=(round(pd_*0.995,2),
                                               round(pd_*1.005,2)),
                              d_level=round(pd_,2), detected=True)

        if best:
            best["harmonic_score"] = int(best["quality"] * 0.8)
            out.update(best)
    except Exception:
        pass
    return out

# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 4 — ADAPTIVE REGIME SCORING
# ══════════════════════════════════════════════════════════════════════════════

_REGIME_WEIGHTS = {
    "TREND_BULL":   dict(TREND=35, MOMENTUM=22, STRUCTURE=18, VOLUME=13, QUALITY=12),
    "TREND_BEAR":   dict(TREND=28, MOMENTUM=15, STRUCTURE=22, VOLUME=18, QUALITY=17),
    "RANGE_BULL":   dict(TREND=22, MOMENTUM=18, STRUCTURE=28, VOLUME=15, QUALITY=17),
    "RANGE_BEAR":   dict(TREND=18, MOMENTUM=12, STRUCTURE=30, VOLUME=18, QUALITY=22),
    "HIGHVOL_BULL": dict(TREND=25, MOMENTUM=18, STRUCTURE=20, VOLUME=17, QUALITY=20),
    "HIGHVOL_BEAR": dict(TREND=15, MOMENTUM=10, STRUCTURE=20, VOLUME=20, QUALITY=35),
    "DEFAULT":      dict(TREND=30, MOMENTUM=20, STRUCTURE=20, VOLUME=15, QUALITY=15),
}
_REGIME_LABELS = {
    "TREND_BULL": "Trending Bull",   "TREND_BEAR": "Trending Bear",
    "RANGE_BULL": "Ranging Bull",    "RANGE_BEAR": "Ranging Bear",
    "HIGHVOL_BULL":"High-Vol Bull",  "HIGHVOL_BEAR":"High-Vol Bear",
    "DEFAULT":    "Default",
}

def classify_regime(market_bullish: bool,
                    adx_val: float,
                    vix_val: float | None = None) -> tuple:
    """
    Classify the market regime into one of 6 buckets.
    Returns (regime_key, regime_label, weights_dict).
    """
    trending = adx_val >= 22
    high_vol  = vix_val is not None and vix_val >= VIX_CAUTION

    if   high_vol  and     market_bullish: key = "HIGHVOL_BULL"
    elif high_vol  and not market_bullish: key = "HIGHVOL_BEAR"
    elif trending  and     market_bullish: key = "TREND_BULL"
    elif trending  and not market_bullish: key = "TREND_BEAR"
    elif market_bullish:                   key = "RANGE_BULL"
    else:                                  key = "RANGE_BEAR"

    return key, _REGIME_LABELS[key], _REGIME_WEIGHTS[key]

# ══════════════════════════════════════════════════════════════════════════════
# ENGINE 5 — CANDLE STRUCTURE INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════

def detect_candle_structure(df: pd.DataFrame,
                             atr_val: float,
                             mode: str = "Swing") -> dict:
    """
    Identify single, two, and three-candle patterns on the last 1–5 bars.
    Returns candle_score (−10 to +10), list of pattern names, and signal.
    """
    out = dict(candle_score=0, patterns=[], candle_signal="NEUTRAL",
               nr7=False, inside_bar=False)
    try:
        if len(df) < 10 or atr_val <= 0:
            return out
        op  = (df["Open"].values.astype(np.float64)
               if "Open" in df.columns else df["Close"].values.astype(np.float64))
        hi  = df["High"].values.astype(np.float64)
        lo  = df["Low"].values.astype(np.float64)
        cl  = df["Close"].values.astype(np.float64)
        n   = len(cl)

        score    = 0
        patterns: list = []

        c0,o0,h0,l0 = cl[-1],op[-1],hi[-1],lo[-1]
        c1,o1,h1,l1 = cl[-2],op[-2],hi[-2],lo[-2]
        body0       = abs(c0 - o0)
        body1       = abs(c1 - o1)
        rng0        = h0 - l0
        bull0       = c0 > o0
        bull1       = c1 > o1
        uw0         = h0 - max(c0, o0)   # upper wick
        lw0         = min(c0, o0) - l0   # lower wick

        # ── Single-candle ──────────────────────────────────────────────────────
        if rng0 > atr_val * 0.5:
            if lw0 > body0 * 2.0 and uw0 < body0 * 0.5:
                patterns.append("Hammer"); score += 4
            if uw0 > body0 * 2.0 and lw0 < body0 * 0.5:
                if bull0: patterns.append("Inverted Hammer"); score += 2
                else:     patterns.append("Shooting Star");  score -= 5

        if rng0 > 0 and body0 / rng0 < 0.10:
            if   lw0 > rng0 * 0.60: patterns.append("Dragonfly Doji");  score += 3
            elif uw0 > rng0 * 0.60: patterns.append("Gravestone Doji"); score -= 3
            else:                   patterns.append("Doji")

        if body0 > atr_val * 0.7:
            if bull0: patterns.append("Bull Marubozu"); score += 3
            else:     patterns.append("Bear Marubozu"); score -= 3

        # ── Two-candle ────────────────────────────────────────────────────────
        if n >= 2:
            if not bull1 and bull0 and body0 > body1*1.2 and c0>o1 and o0<c1:
                patterns.append("Bullish Engulfing"); score += 6
            if bull1 and not bull0 and body0 > body1*1.2 and c0<o1 and o0>c1:
                patterns.append("Bearish Engulfing"); score -= 6
            if h0 < h1 and l0 > l1:
                patterns.append("Inside Bar"); out["inside_bar"] = True; score += 1
            if not bull1 and bull0 and o0 < l1 and c0 > (o1+c1)/2:
                patterns.append("Piercing Line"); score += 4
            if bull1 and not bull0 and o0 > h1 and c0 < (o1+c1)/2:
                patterns.append("Dark Cloud Cover"); score -= 4

        # ── Three-candle ──────────────────────────────────────────────────────
        if n >= 3:
            c2,o2 = cl[-3],op[-3]
            bull2 = c2 > o2
            if not bull2 and abs(c1-o1)<atr_val*0.3 and bull0 and c0>(c2+o2)/2:
                patterns.append("Morning Star"); score += 7
            if bull2 and abs(c1-o1)<atr_val*0.3 and not bull0 and c0<(c2+o2)/2:
                patterns.append("Evening Star"); score -= 7
            if bull0 and bull1 and bull2 and c0>c1>c2 and o0>o1>o2:
                patterns.append("3 White Soldiers"); score += 5
            if not bull0 and not bull1 and not bull2 and c0<c1<c2 and o0<o1<o2:
                patterns.append("3 Black Crows"); score -= 5

        # ── NR7 ───────────────────────────────────────────────────────────────
        if n >= 7:
            ranges = hi[-7:] - lo[-7:]
            if rng0 == float(np.min(ranges)):
                patterns.append("NR7"); out["nr7"] = True; score += 2

        score = int(np.clip(score, -10, 10))
        if   score >=  4: sig = "BULL"
        elif score >=  1: sig = "BULL LEAN"
        elif score <= -4: sig = "BEAR"
        elif score <= -1: sig = "BEAR LEAN"
        else:             sig = "NEUTRAL"

        out.update(candle_score=score, patterns=patterns, candle_signal=sig)
    except Exception:
        pass
    return out

# FIX-B: gate constants
PATTERN_ENRICH_SCORE_MIN = 45
PATTERN_ENRICH_PHASES    = {"ENTRY", "CONT", "BREAKOUT", "SETUP"}

_EMPTY_PAT = dict(
    vcp=dict(detected=False,n_contractions=0,tightest_pct=0.0,vcp_grade="NONE"),
    avwap=dict(avwap=None,price_above=False,near_support=False,pct_above=0.0),
    fib_quality=dict(quality=0,grade="POOR",depth_ok=False,vol_ok=False,recovery_ok=False,fib_level="—"),
    vol_dryup=dict(dry_up=False,intensity=0,bars=0,vol_pct=100.0),
    rel_vol=dict(rel_vol_pct=50.0,label="NORMAL",ratio=1.0),
    darvas=dict(in_box=False,breakout=False,box_top=0.0,box_bottom=0.0,box_width_pct=0.0,bars_in_box=0),
    total_pattern_pts=0)

def _apply_pattern_keys(r: dict, patterns: dict) -> dict:
    r["Patterns"]     = patterns
    r["VCP"]          = patterns.get("vcp",{}).get("detected",False)
    r["VCPGrade"]     = patterns.get("vcp",{}).get("vcp_grade","NONE")
    r["AVWAP"]        = patterns.get("avwap",{}).get("avwap")
    r["AVWAPAbove"]   = patterns.get("avwap",{}).get("price_above",False)
    r["FibQuality"]   = patterns.get("fib_quality",{}).get("quality",0)
    r["FibGrade"]     = patterns.get("fib_quality",{}).get("grade","POOR")
    r["VolDryup"]     = patterns.get("vol_dryup",{}).get("dry_up",False)
    r["VDUIntensity"] = patterns.get("vol_dryup",{}).get("intensity",0)
    r["RVolPct"]      = patterns.get("rel_vol",{}).get("rel_vol_pct",50.0)
    r["RVolLabel"]    = patterns.get("rel_vol",{}).get("label","NORMAL")
    r["RVOL"]         = patterns.get("rel_vol",{}).get("ratio",1.0)
    r["DarvasIn"]     = patterns.get("darvas",{}).get("in_box",False)
    r["DarvasBrk"]    = patterns.get("darvas",{}).get("breakout",False)
    r["DarvasTop"]    = patterns.get("darvas",{}).get("box_top",0.0)
    return r

def enrich_with_patterns(results: list, data: dict, mode: str, market_open: bool) -> list:
    """FIX-B: pattern engine runs ONLY on shortlisted stocks after Stage-B."""
    to_enrich = [r for r in results
                 if r.get("Score",0) >= PATTERN_ENRICH_SCORE_MIN
                 and r.get("Phase","") in PATTERN_ENRICH_PHASES
                 and r.get("Symbol","") in data
                 and data.get(r.get("Symbol","")) is not None
                 and not data.get(r.get("Symbol",""),pd.DataFrame()).empty]
    passthrough = [r for r in results if r not in to_enrich]
    for r in passthrough:
        _apply_pattern_keys(r, dict(_EMPTY_PAT))

    def _enrich_one(r):
        sym = r["Symbol"]; df = data[sym]; atr_val = r.get("ATR", 0.0)
        try:
            pts, patterns = score_all_patterns(df, atr_val=atr_val, mode=mode,
                                               market_open=market_open)
        except Exception:
            pts, patterns = 0, dict(_EMPTY_PAT)
        _apply_pattern_keys(r, patterns)
        if pts > 0:
            # FIX-5: scale down pattern bonus when exhaustion flags have fired
            # so patterns cannot leapfrog a stock past the exhaustion gate
            ext_n = r.get("ExtN", 0)
            if ext_n >= 3:
                bonus = 0.0                     # critical exhaustion — block bonus entirely
            elif ext_n == 2:
                bonus = round(pts * 0.25, 1)    # moderate exhaustion — halved bonus
            else:
                bonus = round(pts * 0.5, 1)     # clean stock — full 50% bonus
            r["Score"] = round(min(100.0, r.get("Score", 0) + bonus), 1)
            ns = r["Score"]
            r["Action"] = ("STRONG BUY" if ns>=75 else "BUY" if ns>=58
                           else "WATCH" if ns>=42 else "SKIP")
        return r

    if to_enrich:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(16,len(to_enrich))) as pool:
            enriched = list(pool.map(_enrich_one, to_enrich))
    else:
        enriched = []
    all_out = enriched + passthrough
    all_out.sort(key=lambda x: x.get("Score",0), reverse=True)
    return all_out

# ══════════════════════════════════════════════════════════════════════════════
# FIX-C: CATEGORY-BASED WEIGHTED SCORING (v15.3 calibrated)
# ══════════════════════════════════════════════════════════════════════════════

_CAT_W = dict(TREND=30, MOMENTUM=20, STRUCTURE=20, VOLUME=15, QUALITY=15)
_PHASE_RAW = {"BREAKOUT":100,"CONT":85,"ENTRY":65,"SETUP":40,"IDLE":10,"EXIT":0}

