"""
scanner.py — run_scan (long), short scan, exit-candidate derivation.
"""
import time
import json
import threading
import concurrent.futures
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config import (
    MODE_CFG, VIX_CALM, VIX_CAUTION, VIX_STRESS,
    LIQUIDITY_MIN_CR, PHASE_BRK, PHASE_CONT, PHASE_ENTRY, PHASE_SETUP, PHASE_IDLE,
    SHORT_SKIP, SHORT_WATCH, SHORT_SIGNAL, SHORT_CONFIRMED,
    SHORT_SCORE_WATCH, SHORT_SCORE_SIGNAL, SHORT_SCORE_CONFIRMED,
    SHORT_HARD_WEIGHT, SHORT_SOFT_WEIGHT,
    _SECTORS_LOOKUP,
)
from universe import SECTOR_MAP
from data_fetch import (
    batch_incremental_fetch, fetch_async, prefetch_htf_parallel, fetch_vix,
    fetch_nifty, fetch_indices, _cold_start_needed, _mark_cold_start_done,
)
from indicators import (
    stage_a_prefilter, ema, rsi, atr_series, to_nse,
    action_label, action_label_with_preconfirm,
)
from scoring import (
    score_stock, record_phase_transition, phase_transition_conf_bonus,
    compute_readiness_score, compute_trade_intent, get_failure_confidence_penalty,
    compute_smart_money_model, compute_accumulation_sequence,
)
from patterns import prefetch_mtf_parallel, enrich_with_patterns
from market import (
    compute_market_regime, get_cached_regime, compute_breadth,
    compute_rs_ranks, _52w_return, compute_rs_leadership,
)
from risk import (
    liquidity_ok, signal_is_stale, vix_target_mult,
)
from persistence import _db_conn, _db_ensure, _db_save, _db_load, _get_earnings_cached

def run_scan(symbols, mode, progress_bar, status_text,
             vix_val=None, min_liq_cr=LIQUIDITY_MIN_CR):

    cfg      = MODE_CFG[mode]
    min_bars = cfg["hist_min_bars"]
    total    = len(symbols)

    # ── v16.1: Use precomputed regime (cached 1 h) — avoids a redundant Nifty fetch ──
    market_bullish, regime_label, _ = get_cached_regime(mode)
    nifty = fetch_nifty(mode)   # still needed for per-stock scoring context

    # ── Index data for Dashboard — fetch Nifty + Sensex in ONE async call ────
    _idx_raw = fetch_async(["^NSEI", "^BSESN"], "5d", "1d", concurrency=2)

    def _idx_snap(df):
        """Extract last close, prev close, change% from an index df."""
        if df is None or df.empty or len(df) < 2:
            return None
        try:
            last  = float(df["Close"].iloc[-1])
            prev  = float(df["Close"].iloc[-2])
            chg   = round((last - prev) / prev * 100, 2)
            return {"last": last, "prev": prev, "chg": chg}
        except Exception:
            return None

    st.session_state["index_nifty"]  = _idx_snap(_idx_raw.get("^NSEI",  pd.DataFrame()))
    st.session_state["index_sensex"] = _idx_snap(_idx_raw.get("^BSESN", pd.DataFrame()))

    if not market_bullish:
        st.warning(
            f"⚠️ **Market Regime: {regime_label}** — EMA20 below EMA50. "
            "Scores haircut 15%. Targets compressed."
        )

    # ── SPEED-3: incremental batch fetch ──────────────────────────────────
    cold  = _cold_start_needed(mode)
    status_text.text(
        f"{'🌅 Cold-start: full fetch' if cold else '⚡ Incremental: live tail'} "
        f"for {total} symbols…"
    )
    progress_bar.progress(0.05)

    data = batch_incremental_fetch(
        symbols, mode, force_full=cold,
        progress_cb=lambda p: progress_bar.progress(0.05 + p*0.30),
    )

    if cold:
        _mark_cold_start_done(mode)

    progress_bar.progress(0.35)

    # ── SPEED-2: Stage-A pre-filter ────────────────────────────────────────
    status_text.text("⚡ Stage-A: fast EMA pre-filter…")
    valid_data  = {s: df for s,df in data.items()
                   if df is not None and not df.empty and len(df) >= min_bars}
    survivors   = stage_a_prefilter(valid_data, mode, min_bars=min_bars)

    # ── PORTFOLIO BYPASS: always score held positions regardless of Stage-A ──
    # Portfolio stocks may be pulling back (price below EMAs) and would be
    # silently dropped before scoring, meaning exit signals are never generated.
    # Force-include any portfolio symbol that has valid data but was filtered out.
    _portfolio_syms = {
        p["symbol"]
        for p in (st.session_state.get("open_positions") or [])
        if isinstance(p, dict) and p.get("symbol")
    }
    _survivors_set  = set(survivors)
    _pf_rescued = [
        sym for sym in _portfolio_syms
        if sym in valid_data and sym not in _survivors_set
    ]
    if _pf_rescued:
        survivors = survivors + _pf_rescued

    rejected    = total - len(survivors)

    progress_bar.progress(0.40)
    _rescued_note = f" · {len(_pf_rescued)} portfolio rescued" if _pf_rescued else ""
    status_text.text(f"Stage-A: {len(survivors)}/{total} survive "
                     f"({rejected} filtered{_rescued_note}) → Stage-B…")

    # ── HTF pre-fetch (survivors only — much smaller set) ─────────────────
    htf_map = prefetch_htf_parallel(survivors, mode, status_text, progress_bar)
    progress_bar.progress(0.55)

    # ── MTF pre-fetch (Engine 1 — secondary/tertiary TFs for survivors) ───
    status_text.text("📊 Multi-timeframe data for survivors…")
    mtf_prefetched = prefetch_mtf_parallel(survivors, mode)
    progress_bar.progress(0.57)

    # ── RS ranks ──────────────────────────────────────────────────────────
    sym_52w_returns = {sym: _52w_return(valid_data[sym]["Close"])
                       for sym in survivors if sym in valid_data}
    rs_rank_map     = compute_rs_ranks(sym_52w_returns)

    phase_history_snapshot = dict(st.session_state.get("phase_history", {}))

    # ── SPEED-2: Stage-B full scoring ─────────────────────────────────────
    status_text.text(f"🔬 Stage-B: full scoring {len(survivors)} survivors…")
    results     = []
    liq_skipped = 0
    n_surv      = len(survivors)
    scored      = 0

    # Fetch daily context for Intraday survivors
    daily_closes: dict = {}
    if mode == "Intraday" and survivors:
        daily_cfg = MODE_CFG["Swing"]
        d_raw     = fetch_async(survivors, daily_cfg["yf_period"],
                                daily_cfg["interval"], concurrency=20)
        for sym, df in d_raw.items():
            if df is not None and not df.empty:
                daily_closes[sym] = df["Close"]

    def _score_one(sym):
        df        = valid_data[sym]
        htf_up, _ = htf_map.get(sym, (True, "HTF-UNKNOWN"))
        rs_rank   = rs_rank_map.get(sym, 50)
        return sym, score_stock(
            df, nifty, mode,
            daily_close            = daily_closes.get(sym),
            market_bullish         = market_bullish,
            vix_val                = vix_val,
            min_liquidity_cr       = min_liq_cr,
            sym                    = sym,
            htf_up                 = htf_up,
            rs_rank                = rs_rank,
            phase_history_snapshot = phase_history_snapshot,
            mtf_prefetched         = mtf_prefetched.get(sym, {}),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, n_surv)) as pool:
        futures = {pool.submit(_score_one, sym): sym for sym in survivors}
        for fut in concurrent.futures.as_completed(futures):
            sym, res = fut.result()
            scored  += 1
            progress_bar.progress(0.55 + scored/max(n_surv,1)*0.40)
            if scored % 20 == 0:
                status_text.text(f"Stage-B: scored {scored}/{n_surv}…")
            if res:
                res["Regime"] = regime_label
                res["Symbol"] = sym
                res["Sector"] = SECTOR_MAP.get(sym) or _SECTORS_LOOKUP.get(sym, "Other")
                res["PortfolioBypass"] = sym in _pf_rescued
                if not res["LiquidityOK"]:
                    liq_skipped += 1
                results.append(res)

    for res in results:
        sym   = res["Symbol"]
        phase = res["_detected_phase"]
        record_phase_transition(sym, phase)
        res["PhaseBonus"] = phase_transition_conf_bonus(sym)

    # FIX-B / v16.1: pattern enrichment runs in a background thread so the
    # scan returns immediately with core scores. The UI shows a "⏳ Enriching…"
    # badge and the next auto-rerun (st.rerun after scan) picks up the result.
    _ENRICH_CACHE = _CACHE_DIR / "enrichment_pending.json"

    def _run_enrichment_bg(results_snapshot, valid_data_snapshot, mode_snap, market_open_snap):
        """Write enriched results to disk; main thread picks up on next rerun."""
        try:
            enriched = enrich_with_patterns(
                results_snapshot, data=valid_data_snapshot,
                mode=mode_snap, market_open=market_open_snap
            )
            with open(_ENRICH_CACHE, "w") as f:
                json.dump(enriched, f, default=str)
        except Exception:
            pass

    import threading as _threading
    _bg = _threading.Thread(
        target=_run_enrichment_bg,
        args=(list(results), dict(valid_data), mode, _is_market_open()),
        daemon=True,
    )
    _bg.start()
    st.session_state["enrichment_ready"] = False  # flag for UI badge

    breadth_pulse = compute_breadth(results)
    pct_ema50_now = breadth_pulse.get("pct_above_ema50", 100)
    ad_ratio_now  = breadth_pulse.get("ad_ratio", 2.0)

    # ── v15.5: Enrich Emerging Scores with sector momentum ────────────────
    _sec_avg   = breadth_pulse.get("sector_avg", {})
    _ovl_avg   = float(np.mean(list(_sec_avg.values()))) if _sec_avg else 50.0
    for _res in results:
        _sb = 0.0
        if _sec_avg:
            _sa = _sec_avg.get(_res.get("Sector", "Other"), _ovl_avg)
            if   _sa >= _ovl_avg + 10: _sb = 10.0
            elif _sa >= _ovl_avg + 5:  _sb = 7.0
            elif _sa >= _ovl_avg:      _sb = 4.0
            elif _sa >= _ovl_avg - 5:  _sb = 2.0
        _res["EmSectorMom"] = round(_sb, 1)
        _em_total = round(min(100.0, _res.get("EmScore", 0) + _sb), 1)
        _res["EmScore"] = _em_total
        _res["EmLabel"] = ("IGNITING" if _em_total >= 65 else "BUILDING" if _em_total >= 50
                           else "COILING" if _em_total >= 35 else "LATENT" if _em_total >= 20
                           else "QUIET")
    # ─────────────────────────────────────────────────────────────────────────

    # ── v15.9: Post-hoc cross-field enrichment — single-pass, cached lookups ──
    # Resolves circular dependencies: PCA→SmartMoney→AccumSeq→RSLead→PRE-CONFIRM
    # Each function called ONCE per stock with real values. No duplicate calls.
    _sec_avg_2  = breadth_pulse.get("sector_avg", {})
    _ovl_avg_2  = float(np.mean(list(_sec_avg_2.values()))) if _sec_avg_2 else 50.0
    _vd_cache   = valid_data  # already in memory — no re-fetch needed

    for _res in results:
        _sym  = _res.get("Symbol", "")
        _pca  = _res.get("PCAScore", 0.0)
        _em   = _res.get("EmScore",  0.0)
        _ph   = _res.get("Phase",    PHASE_IDLE)
        _sec  = _res.get("Sector",   "Other")
        _sec_score = _sec_avg_2.get(_sec, _ovl_avg_2)
        # Retrieve df once — used by all three re-computations below
        _df   = _vd_cache.get(_sym, pd.DataFrame())

        # Step 1: SmartMoney (needs real PCA)
        _sm = compute_smart_money_model(
            _df, mode,
            pca_score  = _pca,
            inst_score = _res.get("InstScore", 50.0),
            obv_trend  = _res.get("InstOBV",   True),
        )
        _res.update(_sm)

        # Step 2: AccumSequence (needs real PCA + Em + SmartMoney from Step 1)
        _as = compute_accumulation_sequence(
            _df, mode,
            pca_score           = _pca,
            em_score            = _em,
            phase               = _ph,
            smart_money_verdict = _res.get("SmartMoneyVerdict", "NEUTRAL"),
            rs_line_high        = _res.get("RSLineHigh", False),
        )
        _res.update(_as)

        # Step 3: RSLeadership (needs real sector avg — no df re-fetch)
        _close_ser = _df["Close"] if not _df.empty else pd.Series(dtype=float)
        _rl = compute_rs_leadership(
            close            = _close_ser,
            nifty_close      = nifty,
            rs_rank          = _res.get("RS_Rank", 50),
            sector_avg_score = _sec_score,
            stock_score      = _res.get("Score", 50.0),
        )
        _res.update(_rl)

        # Step 4: PRE-CONFIRM action tier (all upstream values now resolved)
        _cur_action = _res.get("Action", "SKIP")
        if _cur_action not in ("STRONG BUY", "BUY"):
            _pc_action = action_label_with_preconfirm(
                norm_score          = _res.get("Score", 0.0),
                pca_score           = _pca,
                em_score            = _em,
                phase               = _ph,
                smart_money_verdict = _res.get("SmartMoneyVerdict", "NEUTRAL"),
                accum_stage         = _res.get("AccumStage", "NONE"),
            )
            _res["Action"] = _pc_action

        # ── FIX-6: AccumStage phase override — reduce pure price dependency ──────
        # Structural base evidence (Wyckoff/Weinstein stage) can promote phase
        # even when price hasn't crossed any EMA yet.
        _ast_now = _res.get("AccumStage", "NONE")
        _ph_now  = _res.get("Phase", PHASE_IDLE)
        if _ast_now in ("1A", "1B") and _ph_now == PHASE_IDLE:
            # Clear structural base detected → upgrade to SETUP
            _res["Phase"] = PHASE_SETUP
        elif _ast_now in ("1C", "2A") and _ph_now in (PHASE_IDLE, PHASE_SETUP):
            # Spring / early markup confirmed → upgrade to ENTRY
            smv_now = _res.get("SmartMoneyVerdict", "NEUTRAL")
            if smv_now in ("ACCUMULATING", "MARKUP_READY", "ABSORBING"):
                _res["Phase"] = PHASE_ENTRY

    breadth_weak  = (pct_ema50_now < 40) and (ad_ratio_now < 0.8)

    if breadth_weak:
        gated_count = 0
        for res in results:
            if res.get("Phase") in (PHASE_BRK, PHASE_CONT):
                if res["Action"] in ("STRONG BUY","BUY"):
                    res["Action"]       = "WATCH"
                    res["BreadthGated"] = True
                    gated_count        += 1
            # v15.7: PRE-CONFIRM is never breadth-gated — these stocks haven't
            # confirmed yet, so breadth doesn't apply to them.
            # (PRE-CONFIRM only fires on SETUP/IDLE phase anyway)
        if gated_count:
            st.warning(
                f"⚠️ **Breadth Gate active** — only {pct_ema50_now}% above EMA50, "
                f"A/D ratio {ad_ratio_now:.2f}. "
                f"{gated_count} BREAKOUT/CONT signals capped to WATCH."
            )

    # ── MASTER REGIME (computed after breadth is known) ──────────────────
    _master_regime_result = compute_market_regime(
        vix_val         = vix_val,
        pct_above_ema50 = breadth_pulse.get("pct_above_ema50", 50.0),
        ad_ratio        = breadth_pulse.get("ad_ratio", 1.0),
        pct_advancing   = breadth_pulse.get("pct_advancing", 50.0),
        pct_breakout    = breadth_pulse.get("pct_breakout", 2.0),
        nifty_close     = nifty,
    )
    _master_regime      = _master_regime_result["regime"]
    _regime_adjustments = _master_regime_result["adjustments"]
    st.session_state["master_regime"] = _master_regime_result

    # ── v16.0 FIX-1/3/7: ReadinessScore, PCAPercentile, TradeIntent ────────────
    # FIX-3: PCA percentile — rank within THIS scan universe for context
    _pca_vals  = [r.get("PCAScore", 0.0) for r in results]
    _pca_sorted = sorted(_pca_vals)
    _n_pca     = max(len(_pca_sorted), 1)

    for _res in results:
        # ── 3-AXIS READINESS: Cause + Timing + Context (with ortho penalty) ─
        _scores = compute_readiness_score(
            score                = _res.get("Score", 0.0),
            pca_score            = _res.get("PCAScore", 0.0),
            em_score             = _res.get("EmScore", 0.0),
            rs_leader_score      = float(_res.get("RSLeaderScore", _res.get("RS_Rank", 50.0))),
            mtf_score            = _res.get("MTFScore", 50.0),
            smart_money_score    = _res.get("SmartMoneyScore", 50.0),
            accum_sequence_score = _res.get("AccumSequenceScore", 50.0),
            micro_score          = _res.get("MicroScore", 50.0),
            sqz_bars             = _res.get("SqzBars", 0),
            regime               = _master_regime,
        )
        _res.update(_scores)

        # ── OUTCOME FEEDBACK: reduce confidence for recently-failed setups ─
        _fail_pen = get_failure_confidence_penalty(
            sym        = _res.get("Symbol", ""),
            setup_type = _res.get("Setup", "default"),
            sector     = _res.get("Sector", "Other"),
        )
        if _fail_pen > 0:
            _res["Confidence"]    = max(0, _res.get("Confidence", 50) - int(_fail_pen))
            _res["FailurePenalty"] = round(_fail_pen, 1)

        # Regime adjustment on score threshold: demote Action if below floor
        _score_floor = _regime_adjustments.get("score_floor", 50)
        if _res.get("Score", 0) < _score_floor and _res.get("Action") in ("BUY", "STRONG BUY"):
            _res["Action"] = "WATCH"
            _res["RegimeGated"] = True
        # FIX-3: PCAPercentile — relative rank in this universe (more meaningful than raw)
        _pca_raw = _res.get("PCAScore", 0.0)
        _rank    = sum(1 for v in _pca_sorted if v <= _pca_raw)
        _res["PCAPercentile"] = round(_rank / _n_pca * 100)

        # FIX-7: TradeIntent — unambiguous plain-English decision
        _res.update(compute_trade_intent(
            action       = _res.get("Action", "SKIP"),
            ext_n        = _res.get("ExtN", 0),
            breadth_gated= _res.get("BreadthGated", False),
        ))

    progress_bar.progress(1.0)
    results.sort(key=lambda x: x["Score"], reverse=True)
    return results, rejected, liq_skipped

# ══════════════════════════════════════════════════════════════════════════════
# SHORT SELL ENGINE (unchanged logic, uses new fetch path)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ShortResult:
    symbol:str; verdict:str=SHORT_SKIP; short_score:int=0
    hard_triggers:list=field(default_factory=list)
    soft_triggers:list=field(default_factory=list)
    entry_zone_lo:float=0.0; entry_zone_hi:float=0.0
    stop_loss:float=0.0; target1:float=0.0; target2:float=0.0; target3:float=0.0
    risk_reward:float=0.0; current_price:float=0.0; atr:float=0.0
    rsi_val:float=50.0; volume_ratio:float=1.0; rs_rank:int=50
    htf_trend:str="HTF-UNKNOWN"; phase:str=PHASE_IDLE; ext_n:int=0
    sector:str="—"; mode:str="Swing"
    scanned_at:str=field(default_factory=lambda: datetime.now().isoformat())
    error:str=""; day_change:float=0.0

def score_short(sym:str, mode:str="Swing", htf_cache:dict=None,
                rs_ranks:dict=None, vix_val:float=None,
                prefetched_df:pd.DataFrame=None) -> "ShortResult":
    result = ShortResult(symbol=sym, mode=mode, sector=SECTOR_MAP.get(sym,"—"))
    cfg    = MODE_CFG[mode]
    try:
        df = prefetched_df.copy() if (prefetched_df is not None and not prefetched_df.empty) else pd.DataFrame()
        if df.empty:
            raw = fetch_async([sym], cfg["yf_period"], cfg["interval"], concurrency=1)
            df  = raw.get(sym, pd.DataFrame())
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        if len(df) < 60: result.error="insufficient data"; return result

        cl=df["Close"]; hi=df["High"]; lo=df["Low"]; vol=df["Volume"]
        close=float(cl.iloc[-1]); result.current_price=close
        ef_ser=ema(cl,cfg["ema_fast"]); es_ser=ema(cl,cfg["ema_slow"])
        ef=float(ef_ser.iloc[-1]); es=float(es_ser.iloc[-1])
        atr_s=atr_series(df); atr_v=float(atr_s.iloc[-1])
        atr_mean=float(atr_s.rolling(20).mean().iloc[-1]); result.atr=atr_v
        rsi_ser=rsi(cl,cfg["rsi_len"]); rsi_v=float(rsi_ser.iloc[-1])
        result.rsi_val=round(rsi_v,1)
        avg_vol=float(vol.rolling(20).mean().iloc[-1]) or 1
        result.volume_ratio=round(float(vol.iloc[-1])/avg_vol,2)
        def _ret(n):
            if len(cl)<=n: return 0.0
            return float((cl.iloc[-1]-cl.iloc[-n])/cl.iloc[-n]*100)
        mom1=_ret(22); mom3=_ret(66)
        w52_lo=float(lo.iloc[-252:].min()) if len(lo)>=252 else float(lo.min())
        prior_swing_lo=float(lo.iloc[-21:-1].min()) if len(lo)>21 else float(lo.min())
        rsi_5_ago=float(rsi_ser.iloc[-6]) if len(rsi_ser)>=6 else rsi_v
        ticker=to_nse(sym)
        if htf_cache and sym in htf_cache:
            htf_up,htf_label=htf_cache[sym]
        else:
            htf_df=_fetch_htf_cached(ticker,cfg["htf_period"],cfg["htf_interval"],mode=mode)
            htf_up,htf_label=_htf_trend_from_df(htf_df,mode)
        result.htf_trend=htf_label
        rs_rank=rs_ranks.get(sym,50) if rs_ranks else 50; result.rs_rank=rs_rank
        trend_down=close<ef and ef<es; trend_up=close>ef and ef>es
        if trend_down:          phase=PHASE_EXIT
        elif trend_up:          phase=PHASE_CONT if mom1>0 else PHASE_ENTRY
        elif close>ef and ef<es: phase=PHASE_SETUP
        else:                   phase=PHASE_IDLE
        result.phase=phase
        ext_flags,_,_,ext_n=detect_exhaustion(
            close=cl,high=hi,low=lo,volume=vol,rsi_series=rsi_ser,
            e_fast_s=ef_ser,atr_s=atr_s,atr_mean=atr_mean,
            c=close,v=float(vol.iloc[-1]),vol_avg=avg_vol,mode=mode,vix_val=vix_val)
        result.ext_n=ext_n
        score=0; hard_t=[]; soft_t=[]
        if close<ef and ef<es: score+=SHORT_HARD_WEIGHT; hard_t.append("Bearish EMA Stack")
        for i in range(1,min(5,len(ef_ser)-1)+1):
            if (float(ef_ser.iloc[-i])<float(es_ser.iloc[-i]) and
                    float(ef_ser.iloc[-(i+1)])>=float(es_ser.iloc[-(i+1)])):
                score+=SHORT_HARD_WEIGHT; hard_t.append("Death Cross (EMA ×)"); break
        if not htf_up: score+=SHORT_HARD_WEIGHT; hard_t.append(f"HTF Downtrend ({htf_label})")
        near_52w_lo=(close-w52_lo)/w52_lo<0.03 if w52_lo>0 else False
        below_swing=close<prior_swing_lo
        if near_52w_lo or below_swing:
            score+=SHORT_HARD_WEIGHT
            hard_t.append("Near 52W Low" if near_52w_lo else "Below Swing Low")
        if rsi_5_ago>68 and rsi_v<rsi_5_ago-5:
            score+=SHORT_SOFT_WEIGHT; soft_t.append(f"RSI Rollover ({rsi_5_ago:.0f}→{rsi_v:.0f})")
        if rsi_v<42 and not htf_up: score+=SHORT_SOFT_WEIGHT; soft_t.append(f"RSI Bearish Zone ({rsi_v:.0f})")
        if mom1<-cfg["mom1_th"]: score+=SHORT_SOFT_WEIGHT; soft_t.append(f"Neg 1M Mom ({mom1:.1f}%)")
        if mom3<-cfg["mom3_th"]: score+=SHORT_SOFT_WEIGHT; soft_t.append(f"Neg 3M Mom ({mom3:.1f}%)")
        if float(df["Close"].iloc[-1])<float(df["Open"].iloc[-1]) and result.volume_ratio>1.5:
            score+=SHORT_SOFT_WEIGHT; soft_t.append(f"High-Vol Red Day ({result.volume_ratio:.1f}×)")
        _,_,fibs,_=fib_levels(df)
        if fibs:
            if close<fibs.get("618",float("inf")): score+=SHORT_SOFT_WEIGHT; soft_t.append("Below 61.8% Fib")
            elif close<fibs.get("500",float("inf")): score+=SHORT_SOFT_WEIGHT; soft_t.append("Below 50% Fib")
        if rs_rank<30: score+=SHORT_SOFT_WEIGHT; soft_t.append(f"RS Rank Weak ({rs_rank})")
        if vix_val and vix_val>=VIX_STRESS: score+=5
        if ext_n>=2: score+=min(ext_n*4,12)
        result.short_score=min(score,100); result.hard_triggers=hard_t; result.soft_triggers=soft_t
        if score>=SHORT_SCORE_CONFIRMED: result.verdict=SHORT_CONFIRMED
        elif score>=SHORT_SCORE_SIGNAL:  result.verdict=SHORT_SIGNAL
        elif score>=SHORT_SCORE_WATCH:   result.verdict=SHORT_WATCH
        else:                            result.verdict=SHORT_SKIP
        atr_sl_mult=cfg["atr_mult"]*(0.85 if vix_val and vix_val>=VIX_STRESS else 1.0)
        result.entry_zone_lo=round(close,2); result.entry_zone_hi=round(close+atr_v*0.4,2)
        result.stop_loss=round(close+atr_v*atr_sl_mult,2)
        result.target1=round(close-atr_v*cfg["atr_mult"]*1.0,2)
        result.target2=round(close-atr_v*cfg["atr_mult"]*2.0,2)
        result.target3=round(close-atr_v*cfg["atr_mult"]*3.0,2)
        risk=result.stop_loss-close
        result.risk_reward=round((close-result.target2)/risk,2) if risk>0 else 0.0
    except Exception as e:
        result.error=str(e)
    return result

def run_short_scan(symbols,mode,htf_cache=None,rs_ranks=None,
                   vix_val=None,status_text=None,progress_bar=None) -> list:
    total=len(symbols); results=[]; done=0; cfg=MODE_CFG[mode]
    if status_text: status_text.text("Short scan 1/2: Async OHLCV fetch…")
    prefetched = fetch_async(symbols, cfg["yf_period"], cfg["interval"], concurrency=20)
    if progress_bar: progress_bar.progress(0.50)
    if status_text: status_text.text("Short scan 2/2: Scoring…")
    def _one(sym):
        return score_short(sym,mode,htf_cache=htf_cache,rs_ranks=rs_ranks,
                           vix_val=vix_val,prefetched_df=prefetched.get(sym))
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32,total)) as pool:
        futures={pool.submit(_one,sym):sym for sym in symbols}
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result()); done+=1
            if progress_bar: progress_bar.progress(0.50+done/total*0.50)
            if status_text and done%20==0: status_text.text(f"Short scan {done}/{total}…")
    results.sort(key=lambda r:r.short_score,reverse=True)
    return [r for r in results if r.verdict!=SHORT_SKIP and not r.error]

def score_short_from_result(r:dict, mode:str, vix_val:float=None) -> ShortResult:
    sym=r.get("Symbol","")
    result=ShortResult(symbol=sym,mode=mode,sector=r.get("Sector",SECTOR_MAP.get(sym,"—")))
    cfg=MODE_CFG[mode]
    try:
        close=float(r.get("LTP",0))
        if close<=0: result.error="no price"; return result
        atr_v=float(r.get("ATR",0)); rsi_v=float(r.get("RSI",50))
        rs_rank=int(r.get("RS_Rank",50)); ext_n=int(r.get("ExtN",0))
        ext_flags=r.get("ExtFlags",{}); htf_up=bool(r.get("HTFUp",True))
        htf_label="HTF↑" if htf_up else "HTF↓"
        ema_stack=bool(r.get("EMAStack",False)); trend_down=bool(r.get("TrendDown",False))
        trend_up=bool(r.get("TrendUp",False)); fresh_cross=bool(r.get("FreshCross",False))
        mom1=float(r.get("Mom1",0)); mom3=float(r.get("Mom3",0))
        vol_conf=bool(r.get("VolConf",False)); phase=r.get("Phase",PHASE_IDLE)
        chg=float(r.get("%Change",0))
        result.current_price=close; result.atr=atr_v; result.rsi_val=round(rsi_v,1)
        result.rs_rank=rs_rank; result.htf_trend=htf_label; result.phase=phase
        result.ext_n=ext_n; result.day_change=chg
        result.volume_ratio=1.3 if vol_conf else 0.9
        score=0; hard_t=[]; soft_t=[]
        if trend_down: score+=SHORT_HARD_WEIGHT; hard_t.append("Bearish EMA Stack")
        if trend_down and not ema_stack and not htf_up and not fresh_cross:
            score+=SHORT_HARD_WEIGHT; hard_t.append("Bearish EMA Alignment (no golden cross)")
        if not htf_up: score+=SHORT_HARD_WEIGHT; hard_t.append(f"HTF Downtrend ({htf_label})")
        if phase==PHASE_EXIT: score+=SHORT_HARD_WEIGHT; hard_t.append("Phase EXIT (structural downtrend)")
        if ext_flags.get("rsi_overheat") or ext_flags.get("mom_exhaustion"):
            score+=SHORT_SOFT_WEIGHT; soft_t.append("RSI/Mom Exhaustion (ExtFlag)")
        if rsi_v<42 and not htf_up: score+=SHORT_SOFT_WEIGHT; soft_t.append(f"RSI Bearish Zone ({rsi_v:.0f})")
        if mom1<-cfg["mom1_th"]: score+=SHORT_SOFT_WEIGHT; soft_t.append(f"Neg 1M Mom ({mom1:.1f}%)")
        if mom3<-cfg["mom3_th"]: score+=SHORT_SOFT_WEIGHT; soft_t.append(f"Neg 3M Mom ({mom3:.1f}%)")
        if chg<-0.5 and vol_conf: score+=SHORT_SOFT_WEIGHT; soft_t.append(f"High-Vol Red Day ({chg:+.1f}%)")
        if ext_flags.get("bearish_div"): score+=SHORT_SOFT_WEIGHT; soft_t.append("Bearish Divergence (ExtFlag)")
        if rs_rank<30: score+=SHORT_SOFT_WEIGHT; soft_t.append(f"RS Rank Weak ({rs_rank})")
        if vix_val and vix_val>=VIX_STRESS: score+=5
        if ext_n>=2: score+=min(ext_n*4,12)
        result.short_score=min(score,100); result.hard_triggers=hard_t; result.soft_triggers=soft_t
        if score>=SHORT_SCORE_CONFIRMED: result.verdict=SHORT_CONFIRMED
        elif score>=SHORT_SCORE_SIGNAL:  result.verdict=SHORT_SIGNAL
        elif score>=SHORT_SCORE_WATCH:   result.verdict=SHORT_WATCH
        else:                            result.verdict=SHORT_SKIP
        if atr_v>0:
            atr_sl_mult=cfg["atr_mult"]*(0.85 if vix_val and vix_val>=VIX_STRESS else 1.0)
            result.entry_zone_lo=round(close,2); result.entry_zone_hi=round(close+atr_v*0.4,2)
            result.stop_loss=round(close+atr_v*atr_sl_mult,2)
            result.target1=round(close-atr_v*cfg["atr_mult"]*1.0,2)
            result.target2=round(close-atr_v*cfg["atr_mult"]*2.0,2)
            result.target3=round(close-atr_v*cfg["atr_mult"]*3.0,2)
            risk=result.stop_loss-close
            result.risk_reward=round((close-result.target2)/risk,2) if risk>0 else 0.0
    except Exception as e:
        result.error=str(e)
    return result

def derive_short_candidates(scan_results:list, mode:str, vix_val:float=None) -> list:
    out=[]
    for r in scan_results:
        if not r: continue
        sr=score_short_from_result(r,mode,vix_val)
        if sr.verdict!=SHORT_SKIP and not sr.error: out.append(sr)
    out.sort(key=lambda s:s.short_score,reverse=True)
    return out

# ══════════════════════════════════════════════════════════════════════════════
# EXIT ENGINE (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
