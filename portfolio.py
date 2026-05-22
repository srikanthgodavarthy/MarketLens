"""
portfolio.py — Exit scoring, position management, run_exit_scan.
"""
import time
import concurrent.futures
import numpy as np
import pandas as pd
import streamlit as st
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config import (
    MODE_CFG, VIX_CALM, VIX_STRESS,
    EXIT_HOLD, EXIT_WATCH_LBL, EXIT_SIGNAL_LBL, EXIT_CONFIRM_LBL, EXIT_COLORS,
    PHASE_BRK, PHASE_CONT, PHASE_ENTRY, PHASE_SETUP, PHASE_EXIT,
)
from universe import SECTOR_MAP
from data_fetch import fetch_async
from indicators import ema, rsi, atr_series
from scoring import compute_confidence, detect_phase_and_entry
from market import get_cached_regime
from risk import detect_exhaustion, compute_stop_loss, compute_dynamic_targets, vix_target_mult
from persistence import _db_save

class ExitResult:
    symbol:str; verdict:str=EXIT_HOLD; exit_score:int=0
    triggers:list=field(default_factory=list)
    trailing_stop:float=None; current_price:float=0.0
    atr:float=0.0; day_pct:float=0.0; error:str=""
    # ── Quality / Risk / Hold axes ─────────────────────────────────
    quality_score:int=0          # 0-100  — thesis still intact?
    quality_label:str="—"        # STRONG / INTACT / DEGRADED / BROKEN
    risk_score:int=0             # 0-100  — how much risk is baked in?
    risk_label:str="—"           # LOW / MODERATE / HIGH / CRITICAL
    hold_score:int=0             # 0-100  — composite keep confidence
    hold_label:str="—"           # STRONG HOLD / HOLD / REDUCE / EXIT
    r_multiple:float=0.0         # current P&L expressed in R units
    drawdown_from_peak:float=0.0 # % drop from the highest close since entry
    phase:str=""                 # current Weinstein/lifecycle phase
    smart_money_verdict:str=""   # ACCUMULATING / ABSORBING / NEUTRAL / DISTRIBUTING
    rs_label:str=""              # LEADER / IMPROVING / NEUTRAL / LAGGARD
    days_held:int=0

def score_exit(sym:str, entry_price:float, mode:str="Swing", vix_val:float=None,
               entry_date:str=None) -> ExitResult:
    """
    Three-axis portfolio decision engine:
      QUALITY (0-100) — is the original thesis still intact?
      RISK    (0-100) — how much downside risk is currently baked in?
      HOLD    (0-100) — composite keep-vs-exit confidence (higher = keep)

    Verdict is derived from both axes, not just exit-trigger stacking.
    """
    result=ExitResult(symbol=sym); cfg=MODE_CFG[mode]
    try:
        # ── 1. Fetch price data ────────────────────────────────────────────────
        raw=fetch_async([sym],cfg["yf_period"],cfg["interval"],concurrency=1)
        df=raw.get(sym,pd.DataFrame())
        if df.empty:
            import yfinance as yf
            df=yf.download(to_nse(sym),period=cfg["period"],interval=cfg["interval"],
                           auto_adjust=True,progress=False,threads=False)
            if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
            df=df.dropna()
        if len(df)<30: result.error="insufficient data"; return result

        cl=df["Close"]; close=float(cl.iloc[-1]); result.current_price=close
        atr_v=float(atr_series(df).iloc[-1]); result.atr=atr_v
        if len(cl)>=2:
            result.day_pct=round((close-float(cl.iloc[-2]))/float(cl.iloc[-2])*100,2)

        ef=float(ema(cl,cfg["ema_fast"]).iloc[-1])
        es=float(ema(cl,cfg["ema_slow"]).iloc[-1])
        rsi_v=float(rsi(cl,cfg["rsi_len"]).iloc[-1])
        avg_vol=float(df["Volume"].rolling(20).mean().iloc[-1]) or 1
        vol_ratio=float(df["Volume"].iloc[-1])/avg_vol
        pnl_pct=(close-entry_price)/entry_price*100 if entry_price else 0

        # R-multiple (P&L expressed in units of initial risk = 1 ATR)
        initial_risk=atr_v if atr_v>0 else entry_price*0.02
        result.r_multiple=round((close-entry_price)/initial_risk,2) if entry_price else 0.0

        # Drawdown from in-sample peak (proxy for max close since entry)
        result.drawdown_from_peak=round((close-float(cl.rolling(63).max().iloc[-1]))/
                                        float(cl.rolling(63).max().iloc[-1])*100,2) if entry_price else 0.0

        # Days held
        if entry_date:
            try:
                result.days_held=(datetime.now().date()-datetime.fromisoformat(entry_date).date()).days
            except Exception: result.days_held=0

        # Adaptive trailing stop
        base_mult=2.0
        if vix_val and vix_val>=VIX_STRESS:    base_mult=1.5
        elif vix_val and vix_val>=VIX_CAUTION: base_mult=1.75
        if pnl_pct>=20: base_mult*=0.7
        elif pnl_pct>=10: base_mult*=0.85
        result.trailing_stop=round(close-atr_v*base_mult,2)

        # HTF trend
        htf_df=_fetch_htf_cached(to_nse(sym),cfg["htf_period"],cfg["htf_interval"],mode=mode)
        htf_up,_=_htf_trend_from_df(htf_df,mode)

        # ── 2-A. Sub-function quality signals (targeted, not full score_stock) ─
        # Phase — derived from EMA structure; matches detect_phase_and_entry logic
        atr_s   = atr_series(df)
        atr_mean= float(atr_s.rolling(20).mean().iloc[-1])
        ema_down= ef < es and float(ema(cl, cfg["ema_fast"]).iloc[-4]) < float(ema(cl, cfg["ema_slow"]).iloc[-4])
        hi52    = float(df["High"].rolling(min(len(df),252)).max().iloc[-1])
        trend_up=   close > es and ef > es
        trend_down= close < es and ema_down
        brk_zone=   close >= hi52 * 0.97
        trail_level=float(cl.iloc[-10:].max()) - atr_v * 1.5
        trail_break=close < trail_level

        if trend_down and ema_down:            phase = PHASE_EXIT
        elif brk_zone and trend_up:            phase = PHASE_BRK
        elif close > ef and ef > es:           phase = PHASE_CONT if close > float(cl.iloc[-4:-1].max()) else PHASE_ENTRY
        elif close > es:                       phase = PHASE_SETUP
        elif trail_break and trend_up:         phase = PHASE_EXIT
        else:                                  phase = PHASE_IDLE
        if not htf_up and phase in (PHASE_ENTRY, PHASE_CONT, PHASE_BRK):
            phase = PHASE_SETUP
        result.phase = phase

        # Smart Money — direct sub-function call (correct signature)
        sm_result   = {}
        try: sm_result = compute_smart_money_model(df, mode)
        except Exception: pass
        sm_verdict  = sm_result.get("SmartMoneyVerdict", "NEUTRAL")
        result.smart_money_verdict = sm_verdict

        # RS Leadership — needs nifty; fetch once, fall back to NEUTRAL gracefully
        rs_label = "NEUTRAL"
        try:
            nifty_cl = fetch_nifty(mode)
            if len(nifty_cl) >= 20:
                rs_result = compute_rs_leadership(close=cl, nifty_close=nifty_cl)
                rs_label  = rs_result.get("RSLeaderLabel", "NEUTRAL")
        except Exception:
            pass
        result.rs_label = rs_label

        # ── 2. QUALITY AXIS (0-100) ────────────────────────────────────────────
        # Positive evidence the original thesis is still intact
        q=0; quality_notes=[]

        # Trend structure (35 pts)
        if close>ef:          q+=15; quality_notes.append("✓ Above fast EMA")
        if close>es:          q+=10; quality_notes.append("✓ Above slow EMA")
        if ef>es:             q+=10; quality_notes.append("✓ EMA uptrend intact")

        # Phase (20 pts)
        phase_q={"BREAKOUT":20,"CONT":18,"ENTRY":14,"SETUP":8,"IDLE":3,"EXIT":0}.get(phase,5)
        q+=phase_q
        if phase in ("BREAKOUT","CONT","ENTRY"): quality_notes.append(f"✓ Phase {phase}")
        elif phase=="EXIT": quality_notes.append("✗ Phase EXIT")

        # Smart Money (20 pts)
        sm_q={"MARKUP_READY":20,"ACCUMULATING":17,"ABSORBING":12,"NEUTRAL":6,
              "DISTRIBUTING":0}.get(sm_verdict,6)
        q+=sm_q
        if sm_verdict in ("MARKUP_READY","ACCUMULATING","ABSORBING"):
            quality_notes.append(f"✓ SM: {sm_verdict}")
        elif sm_verdict=="DISTRIBUTING": quality_notes.append("✗ SM distributing")

        # RS Leadership (15 pts)
        rs_q={"LEADER":15,"IMPROVING":10,"NEUTRAL":5,"LAGGARD":0}.get(rs_label,5)
        q+=rs_q
        if rs_label in ("LEADER","IMPROVING"): quality_notes.append(f"✓ RS {rs_label}")
        elif rs_label=="LAGGARD": quality_notes.append("✗ RS LAGGARD")

        # HTF confirmation (10 pts)
        if htf_up: q+=10; quality_notes.append("✓ HTF uptrend")
        else:             quality_notes.append("✗ HTF downtrend")

        q=min(q,100)
        result.quality_score=q
        result.quality_label=("STRONG" if q>=75 else "INTACT" if q>=50
                               else "DEGRADED" if q>=30 else "BROKEN")

        # ── 3. RISK AXIS (0-100) ──────────────────────────────────────────────
        # Higher risk = more reason to reduce/exit; NOT the same as exit triggers
        risk=0; risk_notes=[]

        # Drawdown from peak (30 pts)
        dd=abs(result.drawdown_from_peak)
        if dd>=15:   risk+=30; risk_notes.append(f"⚠ {dd:.1f}% off peak")
        elif dd>=8:  risk+=20; risk_notes.append(f"▲ {dd:.1f}% off peak")
        elif dd>=4:  risk+=10; risk_notes.append(f"↓ {dd:.1f}% off peak")

        # P&L vs SL (25 pts)
        if pnl_pct<-8:       risk+=25; risk_notes.append(f"✗ SL breach {pnl_pct:.1f}%")
        elif pnl_pct<-4:     risk+=15; risk_notes.append(f"▲ Near SL {pnl_pct:.1f}%")
        elif pnl_pct<0:      risk+=7;  risk_notes.append(f"↓ Underwater {pnl_pct:.1f}%")

        # RSI extension (15 pts)
        if rsi_v>78:    risk+=15; risk_notes.append(f"⚠ RSI extreme {rsi_v:.0f}")
        elif rsi_v>70:  risk+=8;  risk_notes.append(f"▲ RSI extended {rsi_v:.0f}")

        # Distribution / high-vol down day (15 pts)
        if sm_verdict=="DISTRIBUTING":         risk+=12; risk_notes.append("⚠ SM distributing")
        if (vol_ratio>2.0 and close<float(df["Open"].iloc[-1])):
            risk+=10; risk_notes.append("⚠ High-vol red bar")

        # VIX regime (10 pts)
        if vix_val and vix_val>=VIX_STRESS:    risk+=10; risk_notes.append(f"⚠ VIX stress {vix_val:.0f}")
        elif vix_val and vix_val>=VIX_CAUTION: risk+=5;  risk_notes.append(f"▲ VIX caution {vix_val:.0f}")

        # Fib breakdown (5 pts)
        _,_,fibs,_=fib_levels(df)
        if fibs and close<fibs.get("618",0): risk+=5; risk_notes.append("↓ Below 61.8% Fib")

        risk=min(risk,100)
        result.risk_score=risk
        result.risk_label=("CRITICAL" if risk>=70 else "HIGH" if risk>=45
                           else "MODERATE" if risk>=20 else "LOW")

        # ── 4. HOLD SCORE (0-100) — composite keep confidence ─────────────────
        # High quality + low risk + healthy R = strong hold
        hold=int(quality_score_w(q)*0.55 + (100-risk)*0.30 + r_mult_bonus(result.r_multiple)*0.15)
        hold=max(0,min(hold,100))
        result.hold_score=hold
        result.hold_label=("STRONG HOLD" if hold>=70 else "HOLD" if hold>=50
                           else "REDUCE" if hold>=30 else "EXIT")

        # ── 5. EXIT TRIGGER LAYER (existing signals → exit_score) ────────────
        # This layer is now a subordinate signal, not the sole verdict driver
        score=0; triggers=[]
        if close<ef:  score+=20; triggers.append("Price < Fast EMA")
        if close<es:  score+=20; triggers.append("Price < Slow EMA")
        if rsi_v>78:  score+=20; triggers.append(f"RSI Overbought {rsi_v:.0f}")
        if pnl_pct<-8: score+=20; triggers.append(f"SL Hit {pnl_pct:.1f}%")
        if ef<es:     score+=8;  triggers.append("EMA Bear Cross")
        if rsi_v>70:  score+=8;  triggers.append(f"RSI High {rsi_v:.0f}")
        if vol_ratio>2.0 and close<float(df["Open"].iloc[-1]):
            score+=8; triggers.append("High-Vol Down Day")
        if pnl_pct>30: score+=8; triggers.append(f"Profit {pnl_pct:.1f}% — Consider Locking In")
        if vix_val and vix_val>=VIX_STRESS: score+=8; triggers.append(f"VIX Stress {vix_val}")
        if fibs and close<fibs.get("618",0): score+=8; triggers.append("Below 61.8% Fib")
        if not htf_up: score+=8; triggers.append("HTF Downtrend")
        if sm_verdict=="DISTRIBUTING": score+=15; triggers.append("SM Distributing")
        if result.quality_label=="BROKEN": score+=15; triggers.append("Quality Broken")
        if risk_notes: triggers=risk_notes[:4]+["—"]+triggers[:4]  # surface risk first
        result.exit_score=min(score,100); result.triggers=triggers

        # ── 6. VERDICT — driven by all three axes together ────────────────────
        if result.quality_label=="BROKEN" or result.risk_label=="CRITICAL" or score>=65:
            result.verdict=EXIT_CONFIRM_LBL
        elif result.quality_label=="DEGRADED" or result.risk_label=="HIGH" or score>=40:
            result.verdict=EXIT_SIGNAL_LBL
        elif result.quality_label in ("DEGRADED","INTACT") and risk>=20 or score>=20:
            result.verdict=EXIT_WATCH_LBL
        else:
            result.verdict=EXIT_HOLD

    except Exception as e:
        result.error=str(e)
    return result

def quality_score_w(q:int)->int:
    """Monotone transform so mid-range quality is less forgiving."""
    return int(q**0.9 * 100**0.1)

def r_mult_bonus(r:float)->int:
    """Convert R-multiple to 0-100 hold bonus — capped at +3R."""
    if r>=3:   return 100
    if r>=2:   return 80
    if r>=1:   return 60
    if r>=0:   return 40
    if r>=-1:  return 20
    return 0

def run_exit_scan(positions:list, vix_val:float=None) -> dict:
    out={}
    valid=[p for p in positions if isinstance(p,dict) and p.get("symbol")]
    if not valid: return out
    def _one(pos):
        sym=pos["symbol"]
        return sym,score_exit(sym,pos.get("entry_price",0),pos.get("mode","Swing"),vix_val,
                              entry_date=pos.get("entry_date"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16,len(valid))) as pool:
        futures={pool.submit(_one,pos):pos for pos in valid}
        for fut in concurrent.futures.as_completed(futures):
            try:
                sym,er=fut.result(); out[sym]=er
            except Exception as e:
                pos=futures[fut]; sym=pos.get("symbol","?")
                out[sym]=ExitResult(symbol=sym,error=str(e))
    return out

def add_position(sym:str, entry_price:float, qty:int, mode:str, entry_date:str=None):
    pos=dict(symbol=sym.upper(),entry_price=entry_price,qty=qty,mode=mode,
             entry_date=entry_date or datetime.now().date().isoformat(),current_price=entry_price)
    # FIX-7: deduplicate on (symbol, entry_date, entry_price) so different-price
    # lots entered on the same day are preserved separately.
    existing=[p for p in st.session_state.get("open_positions",[])
              if not (p["symbol"]==sym.upper()
                      and p["entry_date"]==pos["entry_date"]
                      and p["entry_price"]==entry_price)]
    st.session_state["open_positions"]=existing+[pos]
    _db_save("bs_positions",st.session_state["open_positions"])

# ══════════════════════════════════════════════════════════════════════════════
# CARD STYLE HELPERS (unchanged from v14.3)
# ══════════════════════════════════════════════════════════════════════════════

