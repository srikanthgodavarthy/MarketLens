"""
scanner_tab.py — Scanner tab renderer.
"""
import pandas as pd
import streamlit as st
from config import (
    PHASE_BRK, PHASE_CONT, PHASE_ENTRY, PHASE_SETUP, PHASE_IDLE,
    SHORT_CONFIRMED, SHORT_SIGNAL,
)
from data_fetch import fetch_indices, fetch_oi_data
from risk import signal_is_stale
from persistence import _get_earnings_cached
from components import _action_colors, _phase_color, _conf_color
from persistence import _result_hash

def render(all_results, vix_val, vix_label, scan_mode, signal_log):
    """Render this tab. Call inside `with tab_X:`."""
    # v16.1: lazy-load earnings only when this tab renders, not during scan.
    _scanner_syms = tuple(r["Symbol"] for r in st.session_state.get("results", []))
    if _scanner_syms and not st.session_state.get("earnings_map"):
        _syms_key = ",".join(sorted(_scanner_syms))
        st.session_state["earnings_map"] = _get_earnings_cached(_syms_key, _scanner_syms)

    # v16.1: show enrichment badge if background thread hasn't finished yet
    if not st.session_state.get("enrichment_ready", True):
        st.info("⏳ Pattern enrichment running in background — scores will update shortly.")

    indices=fetch_indices(scan_mode)
    oi_nifty=fetch_oi_data("NIFTY")
    oi_banknifty=fetch_oi_data("BANKNIFTY")

    ic1,ic2,ic3=st.columns(3)
    for (name,col,oi_data) in [("Nifty 50",ic1,oi_nifty),("BankNifty",ic2,oi_banknifty),("Sensex",ic3,None)]:
        d=indices.get(name)
        with col:
            if not d:
                st.markdown(f"<div style='color:#cbd5e1;font-size:12px;'>{name}: unavailable</div>",unsafe_allow_html=True)
                continue
            chg_val=d["chg"]; pct_val=d["pct"]; ltp_val=d["value"]
            cs=f"+{pct_val:.2f}%" if chg_val>=0 else f"{pct_val:.2f}%"
            cc="#22c55e" if chg_val>=0 else "#ef4444"
            ar="▲" if chg_val>=0 else "▼"
            act=d["action"]
            score_bar_color=("#f59e0b" if act=="STRONG BUY" else "#22c55e" if act=="BUY"
                             else "#f59e0b" if act=="WATCH" else "#cbd5e1")
            sp=int(min(d["score"],100))
            oi_badge=""
            if oi_data:
                s_label,s_col=_oi_sentiment(oi_data["pcr"])
                pd_=oi_data["max_pain"]-int(ltp_val)
                pa="↑" if pd_>0 else ("↓" if pd_<0 else "=")
                oi_badge=(f'<div style="margin-top:6px;padding:5px 8px;background:#09090f;'
                          f'border-radius:5px;border:1px solid #1e1e40;font-family:JetBrains Mono,monospace;">'
                          f'<span style="color:#cbd5e1;font-size:9px;">PCR </span>'
                          f'<span style="background:{s_col}22;border:1px solid {s_col}44;'
                          f'color:{s_col};padding:1px 5px;border-radius:3px;font-size:9px;font-weight:600;">'
                          f'{oi_data["pcr"]} {s_label}</span>'
                          f'<span style="color:#cbd5e1;font-size:9px;margin-left:6px;">Pain </span>'
                          f'<span style="color:#f59e0b;font-size:9px;font-weight:600;">'
                          f'₹{oi_data["max_pain"]:,} {pa}{abs(pd_):,}</span>'
                          f'<br><span style="color:#ef4444;font-size:9px;">C▶₹{oi_data["call_wall"]:,}  </span>'
                          f'<span style="color:#22c55e;font-size:9px;">P▶₹{oi_data["put_wall"]:,}</span></div>')
            st.markdown(
                f'<div style="background:#111120;border:1px solid #1e1e40;border-radius:10px;padding:14px 16px;">'
                f'<div style="font-family:DM Sans,sans-serif;color:#cbd5e1;font-size:10px;text-transform:uppercase;letter-spacing:1px;">{name}</div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:#e8e8f4;font-size:22px;font-weight:600;margin:4px 0 2px;">{ltp_val:,.1f}</div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:{cc};font-size:12px;">{ar} {cs}</div>'
                f'<div style="margin:8px 0 4px;background:#1e1e40;border-radius:3px;height:3px;">'
                f'<div style="background:{score_bar_color};width:{sp}%;height:3px;border-radius:3px;"></div></div>'
                f'<div style="display:flex;align-items:center;gap:6px;margin-top:4px;">'
                f'<span style="background:{score_bar_color}22;border:1px solid {score_bar_color}44;'
                f'color:{score_bar_color};padding:2px 7px;border-radius:3px;font-size:10px;font-weight:600;">{act}</span>'
                f'<span style="font-family:JetBrains Mono,monospace;color:#3a3a60;font-size:10px;">RSI {d["rsi"]} · {d["trend"]}</span>'
                f'</div>'+oi_badge+'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div style="border-top:1px solid #1e1e40;margin:16px 0;"></div>',unsafe_allow_html=True)

    # ── Apply filters ──────────────────────────────────────────────────────────
    results=list(st.session_state.get("results", []))
    fc1, fc2 = st.columns(2)
    with fc1:
        filter_opt = st.selectbox("Filter", ["BUY + STRONG BUY","STRONG BUY only","WATCH + BUY","PRE-CONFIRM","All Results"], label_visibility="collapsed", key="filter_opt_scanner_tab")
    with fc2:
        search_q = st.text_input("Search symbol", placeholder="e.g. RELIANCE", label_visibility="collapsed")

    if filter_opt=="BUY + STRONG BUY": results=[r for r in results if r["Action"] in ("BUY","STRONG BUY")]
    elif filter_opt=="STRONG BUY only": results=[r for r in results if r["Action"]=="STRONG BUY"]
    elif filter_opt=="WATCH + BUY": results=[r for r in results if r["Action"] in ("WATCH","BUY","STRONG BUY")]
    elif filter_opt=="PRE-CONFIRM": results=[r for r in results if r["Action"]=="PRE-CONFIRM"]
    _phase_filter=st.session_state.get("phase_filter","All Phases")
    if _phase_filter!="All Phases": results=[r for r in results if r.get("Phase")==_phase_filter]
    if not st.session_state.get("show_illiquid",False): results=[r for r in results if r.get("LiquidityOK",True)]
    if search_q: results=[r for r in results if search_q.upper() in r["Symbol"]]

    # ── make_card (v14.3 card UI preserved, v15 adds ADX/Squeeze badges) ──────
    if st.session_state.get("results", []):
        scan_mode_now=st.session_state.scan_mode
        stale_syms=set()
        for entry in st.session_state.get("signal_log", []):
            if signal_is_stale(entry["timestamp"],entry.get("mode",scan_mode_now)):
                stale_syms.add(entry["symbol"])

        # ── v15.8-FIX: helpers for unique signal extraction ──────────────────────
        def _unique_signals(r):
            """Extract top-4 most differentiated, stock-specific signals with actual values."""
            sigs = []
            # Squeeze depth + duration
            if r.get("Squeeze"):
                sb = r.get("SqzBars", 0); sd = r.get("SqzDepth", 1.0)
                sigs.append({"label":f"Squeeze {sb}d","value":f"{int((1-sd)*100)}% tight","color":"#c084fc","rank":90+sb})
            # Volume dry-up
            vdu = r.get("Patterns",{}).get("vol_dryup",{})
            if vdu.get("dry_up") and vdu.get("intensity",0)>=1:
                sigs.append({"label":f"Vol Dry {vdu.get('bars',0)}b","value":f"{vdu.get('vol_pct',0):.0f}% avg","color":"#38bdf8","rank":80+vdu.get("intensity",0)*5})
            # ADX
            adx = r.get("ADX",0)
            if adx >= 20:
                ac = "#22c55e" if adx>=30 else "#f59e0b"
                al = "Very strong" if adx>=40 else "Strong" if adx>=30 else "Building"
                sigs.append({"label":f"ADX {adx:.0f}","value":al,"color":ac,"rank":55+adx})
            # MTF sync
            ms = r.get("MTFScore",50); ml = r.get("MTFLabel","NEUTRAL")
            if ms>=62 or ms<=38:
                mc = "#22c55e" if ms>=62 else "#ef4444"
                sigs.append({"label":f"MTF {ms:.0f}","value":ml,"color":mc,"rank":ms if ms>=50 else 100-ms})
            # Institutional
            iv = r.get("InstVerdict","NEUTRAL"); ic = r.get("InstCMF",0); ins = r.get("InstScore",50)
            if ins>=65 or ins<=35:
                ic2 = "#22c55e" if ins>=65 else "#ef4444"
                sigs.append({"label":f"Inst {ins:.0f}","value":f"CMF {ic:+.3f}","color":ic2,"rank":abs(ins-50)+50})
            # VCP
            vcp = r.get("Patterns",{}).get("vcp",{})
            if vcp.get("detected") and vcp.get("n_contractions",0)>=2:
                nc=vcp.get("n_contractions",0); tp=vcp.get("tightest_pct",0)
                sigs.append({"label":f"VCP {nc}×","value":f"{tp:.1f}% tight","color":"#a78bfa","rank":70+nc*5})
            # AVWAP
            av_d = r.get("Patterns",{}).get("avwap",{}); av=av_d.get("avwap"); pa=av_d.get("pct_above",0)
            if av and av_d.get("price_above"):
                avc = "#38bdf8" if av_d.get("near_support") else "#64748b"
                sigs.append({"label":f"AVWAP ₹{av:,.0f}","value":f"+{pa:.1f}% above","color":avc,"rank":60+(5 if av_d.get("near_support") else 0)})
            # RS rank
            rsr=r.get("RS_Rank",50); rsv=r.get("RS",0)
            if rsr>=75 or rsr<=25:
                rsc="#22c55e" if rsr>=75 else "#ef4444"
                arr="↑" if rsv>=0 else "↓"
                sigs.append({"label":f"RS Rank {rsr}","value":f"{arr} {abs(rsv):.1f}% vs Nifty","color":rsc,"rank":rsr if rsr>=50 else 100-rsr})
            # Harmonic
            if r.get("HarmonicDetected"):
                hp=r.get("HarmonicPattern",""); hd=r.get("HarmonicDir",""); hq=r.get("HarmonicQuality",0)
                hc="#22c55e" if hd=="BULL" else "#ef4444"
                sigs.append({"label":f"{hp}","value":f"{hq}% · {hd}","color":hc,"rank":hq})
            # Candle pattern
            cp=r.get("CandlePatterns",[]); cs_=r.get("CandleScore",0)
            if r.get("CandleSignal") in ("BULL","BULL LEAN") and cp:
                sigs.append({"label":cp[0],"value":f"Score +{cs_}","color":"#86efac","rank":50+cs_*3})
            # Darvas breakout
            dv=r.get("Patterns",{}).get("darvas",{})
            if dv.get("breakout"):
                sigs.append({"label":"Darvas Break","value":f"Box {dv.get('box_width_pct',0):.1f}%","color":"#f87171","rank":88})
            # Fib quality
            fq=r.get("Patterns",{}).get("fib_quality",{})
            if fq.get("quality",0)>=60:
                sigs.append({"label":f"Fib {fq.get('fib_level','—')}","value":f"{fq.get('grade','—')} retracement","color":"#fb923c","rank":fq.get("quality",0)})
            # 1M momentum (specific %)
            m1=r.get("Mom1",0)
            if abs(m1)>=5:
                mc2="#22c55e" if m1>0 else "#ef4444"
                sigs.append({"label":"1M Mom","value":f"{m1:+.1f}%","color":mc2,"rank":40+abs(m1)})
            # Fresh EMA cross
            if r.get("FreshCross"):
                sigs.append({"label":"Golden Cross","value":"EMA cross <5 bars","color":"#fbbf24","rank":82})
            # NR7
            if r.get("NR7"):
                sigs.append({"label":"NR7","value":"Narrowest range 7d","color":"#c084fc","rank":72})
            # Smart money (v15.7)
            smv=r.get("SmartMoneyVerdict",""); sms=r.get("SmartMoneyScore",50)
            if smv in ("ACCUMULATING","MARKUP_READY","ABSORBING"):
                smc="#22c55e" if smv in ("ACCUMULATING","MARKUP_READY") else "#38bdf8"
                sigs.append({"label":f"SM {smv[:6]}","value":f"Score {sms:.0f}","color":smc,"rank":85})
            # Accum stage (v15.7)
            ast=r.get("AccumStage","")
            if ast in ("1C","2A","2B"):
                sigs.append({"label":f"Stage {ast}","value":r.get("AccumStageLabel","")[:18],"color":"#22c55e","rank":88 if ast=="2A" else 78})
            sigs.sort(key=lambda x:x["rank"], reverse=True)
            return sigs[:4]

        def _caution_line(r, group=None):
            ext_n=r.get("ExtN",0); ext_lb=r.get("ExtLabels",[])
            # BreadthGated message already appears in the TradeIntent banner — skip here to avoid duplication
            if r.get("BreadthGated"): return None
            is_breakout = group in ("BREAKOUT_READY", "STRONG_BUY")
            if ext_n>=3 and not is_breakout: return f"{'/ '.join(ext_lb[:2]) or 'Exhaustion'} — avoid entry"
            if ext_n==2 and not is_breakout: return f"{ext_lb[0] if ext_lb else 'Ext'} — halve size"
            if r.get("MTFDiverge"): return "TF divergence — confirm HTF first"
            # For breakout stocks, RSI 60–75 is expected; only warn above 78
            rsi_warn_th = 78 if is_breakout else 73
            if r.get("RSI",50)>=rsi_warn_th: return f"RSI {r.get('RSI',50):.0f} — extended, wait"
            if not r.get("HTFUp",True): return "HTF bearish — reduce size"
            ed = st.session_state.get("earnings_map",{}).get(r.get("Symbol",""))
            if ed: return f"Results {ed} — binary risk"
            return None

        # ══════════════════════════════════════════════════════════════════════
        # v15.9: HUMAN LABEL TRANSLATIONS
        # Internal names → plain English shown in every card
        # ══════════════════════════════════════════════════════════════════════
        _HUMAN_LABELS = {
            # Scores
            "PCAScore":          "Accumulation Strength",
            "EmScore":           "Momentum Build-Up",
            "SmartMoneyVerdict": "Institutional Activity",
            "RSLeaderLabel":     "Market Leadership",
            "MicroLabel":        "Order Flow",
            "AccumStage":        "Base Stage",
            "EmRSAccel":         "Leadership Improving",
            "EmATRCompress":     "Volatility Tightening",
            "EmRVolAccel":       "Volume Building",
            "EmEMAConv":         "Trend Aligning",
            "EmSqzPressure":     "Coil Pressure",
            "EmSectorMom":       "Sector Strength",
            # Verdicts
            "MARKUP_READY":      "Ready to Run ▲",
            "ACCUMULATING":      "Institutions Buying",
            "ABSORBING":         "Buyers Absorbing",
            "NEUTRAL":           "No Clear Bias",
            "DISTRIBUTING":      "Selling Pressure ▼",
            # Stages
            "NONE":  "No Base",
            "1A":    "Building Base",
            "1B":    "Testing Support",
            "1C":    "Ready to Break Out",
            "2A":    "Early Uptrend",
            "2B":    "Trending Up",
            # RS Leadership
            "LEADER":    "Market Leader ⭐",
            "IMPROVING": "Gaining Strength ↑",
            "LAGGARD":   "Underperforming ↓",
            # Flow
            "STRONG_BUY_FLOW":  "Strong Buying",
            "BUY_FLOW":         "More Buyers",
            "NEUTRAL_FLOW":     "Balanced",
            "SELL_FLOW":        "More Sellers",
            "STRONG_SELL_FLOW": "Heavy Selling",
            # MTF
            "BULL SYNC":  "All Timeframes Up ↑",
            "BULL LEAN":  "Mostly Bullish",
            "BEAR SYNC":  "All Timeframes Down",
            "BEAR LEAN":  "Mostly Bearish",
            "DIVERGE":    "Mixed Signals",
            # Institutional
            "INST↑": "Institutions Buying",
            "INST↓": "Institutions Selling",
            "INST~": "Neutral",
        }
        def _hl(key):
            """Translate internal key to human label."""
            return _HUMAN_LABELS.get(key, key)
        # Design: shared core (Price · % chg · Action · Phase · SL) +
        #         dynamic intelligence section swapped per lifecycle group.
        # ══════════════════════════════════════════════════════════════════════

        def _resolve_ui_group(r):
            """Map a result record to one of the six lifecycle UI groups."""
            phase   = r.get("Phase", PHASE_IDLE)
            action  = r.get("Action", "SKIP")
            ext_n   = r.get("ExtN", 0)
            short_s = r.get("ShortScore", 0)
            em_s    = r.get("EmScore", 0)
            pca_s   = r.get("PCAScore", 0)
            accum   = r.get("AccumStage", "NONE")
            score   = r.get("Score", 0)

            if short_s >= SHORT_SCORE_SIGNAL:
                return "SHORT_SELL"
            if ext_n >= 2:
                return "EXTENDED_RISKY"
            if action == "STRONG BUY" and phase in (PHASE_ENTRY, PHASE_CONT, PHASE_BRK):
                return "STRONG_BUY"
            if phase in (PHASE_ENTRY, PHASE_CONT, PHASE_BRK) and action in ("BUY", "PRE-CONFIRM"):
                return "BREAKOUT_READY"
            if em_s >= 35 and phase in (PHASE_IDLE, PHASE_SETUP, PHASE_ENTRY):
                return "EMERGING_MOMENTUM"
            if pca_s >= 30 or accum in ("1A", "1B", "1C", "2A"):
                return "ACCUMULATING"
            return "ACCUMULATING"

        # ── Group meta: label + accent color ──────────────────────────────────
        _UI_GROUP_META = {
            "ACCUMULATING":      ("🛡 Quietly Building",     "#38bdf8"),
            "EMERGING_MOMENTUM": ("🌱 Momentum Building",    "#a78bfa"),
            "BREAKOUT_READY":    ("⚡ Ready to Enter",       "#f59e0b"),
            "STRONG_BUY":        ("🚀 Strong Buy Signal",    "#22c55e"),
            "EXTENDED_RISKY":    ("⚠ Overextended — Caution","#ef4444"),
            "SHORT_SELL":        ("🔻 Bearish Setup",        "#cc2244"),
        }

        def _phase_intel_cell(label, value, color, dim=False):
            """Render a single intelligence cell with consistent styling."""
            bg  = f"{color}18" if not dim else "#37415118"
            brd = f"{color}40" if not dim else "#37415130"
            vc  = f"{color}cc" if not dim else "#47556966"
            return (
                f'<div style="background:{bg};border:1px solid {brd};border-radius:6px;'
                f'padding:5px 7px;min-width:0;">'
                f'<div style="color:#475569;font-size:7.5px;letter-spacing:.05em;'
                f'text-transform:uppercase;margin-bottom:2px;">{label}</div>'
                f'<div style="color:{vc};font-family:JetBrains Mono,monospace;'
                f'font-size:10px;font-weight:700;white-space:nowrap;overflow:hidden;'
                f'text-overflow:ellipsis;">{value}</div>'
                f'</div>'
            )

        def _score_bar_mini(score, max_score, color):
            """Tiny horizontal score bar."""
            pct = min(int(score / max(max_score, 1) * 100), 100)
            return (
                f'<div style="height:3px;border-radius:2px;background:#1e2a3a;margin-top:3px;">'
                f'<div style="height:3px;border-radius:2px;background:{color};width:{pct}%;"></div>'
                f'</div>'
            )

        def _build_phase_intel(r, group=None):
            """
            Build the phase-specific intelligence grid HTML.
            Shows ONLY the indicators relevant to the next decision for each lifecycle stage.
            """
            if group is None:
                group = _resolve_ui_group(r)
            grp_label, grp_color = _UI_GROUP_META.get(group, ("Intelligence", "#64748b"))

            cells_html = ""

            if group == "ACCUMULATING":
                # ── Accumulating: base quality — unified AccumPressure replaces
                # separate PCALabel + SmartMoneyVerdict (FIX-4: overlap resolved) ──
                ast     = r.get("AccumStage", "NONE")
                smv     = r.get("SmartMoneyVerdict", "NEUTRAL")
                base_b  = r.get("AccumBarsInBase", 0)
                rs_sl   = r.get("RSLineSlope", r.get("EmRSAccel", 0))
                aseq    = r.get("AccumSequenceScore", 0)
                em_lbl  = r.get("EmLabel", "QUIET")
                # FIX-3: show PCA as percentile, not raw score
                pca_pct = r.get("PCAPercentile", 0)
                ready   = r.get("ReadinessScore", r.get("Score", 0))

                ast_c  = "#22c55e" if ast in ("2A","2B") else "#38bdf8" if ast in ("1B","1C") else "#f59e0b" if ast == "1A" else "#374151"
                smv_c  = "#22c55e" if smv in ("ACCUMULATING","MARKUP_READY") else "#38bdf8" if smv=="ABSORBING" else "#ef4444" if smv=="DISTRIBUTING" else "#64748b"
                base_c = "#f59e0b" if base_b >= 20 else "#38bdf8" if base_b >= 10 else "#64748b"
                rs_c   = "#22c55e" if rs_sl > 0 else "#ef4444" if rs_sl < 0 else "#64748b"
                pct_c  = "#22c55e" if pca_pct >= 70 else "#38bdf8" if pca_pct >= 40 else "#64748b"
                ready_c= "#22c55e" if ready >= 60 else "#f59e0b" if ready >= 40 else "#64748b"
                em_c_m = {"IGNITING":"#f59e0b","BUILDING":"#22c55e","COILING":"#a78bfa","LATENT":"#38bdf8","QUIET":"#475569"}.get(em_lbl,"#475569")
                _em_friendly_acc = {"IGNITING":"Launching","BUILDING":"Building","COILING":"Tightening","LATENT":"Warming","QUIET":"Dormant"}.get(em_lbl, em_lbl)

                cells_html = (
                    _phase_intel_cell("Momentum",        _em_friendly_acc,                                                          em_c_m, dim=em_lbl=="QUIET")
                    + _phase_intel_cell("Where in Base",  _hl(ast),                                                       ast_c,  dim=ast=="NONE")
                    + _phase_intel_cell("Institutions",   _hl(smv),                                                      smv_c,  dim=smv=="DISTRIBUTING")
                    + _phase_intel_cell("Days Building",  f"{base_b}d" if base_b else "—",                               base_c, dim=base_b < 5)
                    + _phase_intel_cell("Structure",      f"{r.get('StructureScore', ready):.0f}/100",                  ready_c, dim=ready < 30)
                    + _phase_intel_cell("Timing",         f"{r.get('TimingScore', ready):.0f}/100",                     "#a78bfa" if r.get("TimingScore",0)>=40 else "#475569", dim=r.get("TimingScore",0)<20)
                )

            elif group == "EMERGING_MOMENTUM":
                # ── Emerging: coil mechanics — hide ADX/ATR/T1/T2/T3 ─────────────
                em_s     = r.get("EmScore", 0)
                act_em   = r.get("Action", "SKIP")
                sqz      = r.get("Squeeze", False)
                sqz_bars = r.get("SqzBars", 0)
                sqz_dep  = r.get("SqzDepth", 1.0)   # 0=very tight, 1=loose
                rs_ac    = r.get("EmRSAccel", 0)
                sec_m    = r.get("EmSectorMom", 0)
                smv      = r.get("SmartMoneyVerdict", "NEUTRAL")
                sm_conf  = r.get("SMConfidence", 0)
                sm_phase = r.get("SMBehaviorPhase", "")
                pca_s    = r.get("PCAScore", 0)
                em_lbl_i = r.get("EmLabel", "QUIET")
                em_atr   = r.get("EmATRCompress", 0)
                em_ema   = r.get("EmEMAConv", 0)
                em_sqzp  = r.get("EmSqzPressure", 0)
                accum_st = r.get("AccumStage", "NONE")
                accum_bars = r.get("AccumBarsInBase", 0)
                accum_conf = r.get("AccumConfidence", 0)
                micro_s  = r.get("MicroScore", 50)
                micro_lbl= r.get("MicroLabel", "NEUTRAL_FLOW")

                # ── MOMENTUM STATE: show label + strongest driver ─────────────────
                _em_friendly_lbl = {"IGNITING":"Launching","BUILDING":"Building","COILING":"Tightening","LATENT":"Warming","QUIET":"Dormant"}.get(em_lbl_i, em_lbl_i)
                em_state_c = {"IGNITING":"#f59e0b","BUILDING":"#22c55e","COILING":"#a78bfa","LATENT":"#38bdf8","QUIET":"#475569"}.get(em_lbl_i,"#475569")
                _em_drivers = []
                if em_atr >= 10:  _em_drivers.append("ATR")
                if em_ema >= 10:  _em_drivers.append("EMA")
                if em_sqzp >= 10: _em_drivers.append("Sqz")
                _em_sub = "+".join(_em_drivers) if _em_drivers else f"{em_s:.0f}/100"
                _mom_val = f"{_em_friendly_lbl} · {_em_sub}" if _em_sub else _em_friendly_lbl

                # ── EARLY ENTRY: show gap to PRE-CONFIRM ─────────────────────────
                if act_em == "PRE-CONFIRM":
                    _ee_val = f"✓ Stg {accum_st}"
                    pc_c    = "#a78bfa"
                elif pca_s >= 45 and smv in ("ABSORBING","ACCUMULATING","MARKUP_READY"):
                    _ee_val = f"PCA✓ SM✓ ↑phase"
                    pc_c    = "#38bdf8"
                elif pca_s >= 45:
                    _ee_val = f"PCA✓ SM?"
                    pc_c    = "#f59e0b"
                else:
                    _ee_val = f"PCA {pca_s:.0f} low"
                    pc_c    = "#374151"

                # ── PRICE COILING: bars + tightness ──────────────────────────────
                if sqz:
                    _tightness = int((1 - sqz_dep) * 100)
                    _sqz_val   = f"{sqz_bars}d · {_tightness}%"
                    sqz_c      = "#22c55e" if sqz_bars >= 10 else "#a78bfa"
                elif em_sqzp >= 8:
                    _sqz_val   = f"Forming {em_sqzp:.0f}/15"
                    sqz_c      = "#f59e0b"
                else:
                    _sqz_val   = "Not yet"
                    sqz_c      = "#374151"

                # ── INSTITUTIONS: verdict + confidence ───────────────────────────
                smv_c = "#22c55e" if smv in ("ACCUMULATING","MARKUP_READY") else "#38bdf8" if smv=="ABSORBING" else "#f59e0b" if smv=="NEUTRAL" else "#ef4444"
                if smv in ("ACCUMULATING","MARKUP_READY"):
                    _inst_val = f"{_hl(smv)[:8]} {sm_conf}%"
                elif smv == "ABSORBING":
                    _inst_val = f"Absorb {sm_conf}%"
                elif smv == "NEUTRAL":
                    _inst_val = f"Neut · PCA{pca_s:.0f}"
                else:
                    _inst_val = f"DIST ⚠ {sm_conf}%"
                    smv_c     = "#ef4444"

                rs_c   = "#22c55e" if rs_ac >= 8 else "#38bdf8" if rs_ac >= 4 else "#64748b"
                sm_c   = "#22c55e" if sec_m >= 6 else "#38bdf8" if sec_m >= 3 else "#64748b"

                cells_html = (
                    _phase_intel_cell("Momentum State",  _mom_val,       em_state_c, dim=em_lbl_i=="QUIET")
                    + _phase_intel_cell("Early Entry?",  _ee_val,        pc_c,       dim=act_em != "PRE-CONFIRM")
                    + _phase_intel_cell("Price Coiling", _sqz_val,       sqz_c,      dim=not sqz and em_sqzp < 8)
                    + _phase_intel_cell("RS vs Market",  f"+{rs_ac:.0f} accel", rs_c, dim=rs_ac < 3)
                    + _phase_intel_cell("Sector Strength",f"{sec_m:.0f}/10", sm_c,   dim=sec_m < 2)
                    + _phase_intel_cell("Institutions",  _inst_val,      smv_c,      dim=smv=="DISTRIBUTING")
                )

            elif group == "BREAKOUT_READY":
                # ── Breakout Ready: Entry trigger · Vol confirm · HTF · R:R · Confidence · Brk dist ──
                ltp      = r.get("LTP", 0)
                entry    = r.get("Entry") or ltp
                conf     = r.get("Confidence", 0)
                vol_conf = r.get("VolConf", False)
                htf_up   = r.get("HTFUp", True)
                sl       = r.get("SL") or 0
                t1       = r.get("T1") or 0
                rr       = ((t1 - entry) / (entry - sl)) if (entry and sl and t1 and (entry - sl) > 0) else 0
                brk_dist = abs(entry - ltp) / ltp * 100 if ltp else 0
                rvol_brk = r.get("RVOL", 1.0)
                setup_brk = r.get("Setup", "—")

                vol_c    = "#22c55e" if vol_conf else "#ef4444"
                htf_c    = "#22c55e" if htf_up else "#ef4444"
                conf_c   = "#22c55e" if conf >= 70 else "#f59e0b" if conf >= 50 else "#64748b"
                brk_c    = "#22c55e" if brk_dist < 1.0 else "#f59e0b" if brk_dist < 3 else "#ef4444"
                rv_brk_c = "#22c55e" if rvol_brk >= 2.0 else "#f59e0b" if rvol_brk >= 1.3 else "#64748b"
                stp_c    = "#22c55e" if setup_brk in ("breakout","fib") else "#f59e0b"

                rsi_brk   = r.get("RSI", 0)
                em_lbl_brk = r.get("EmLabel", "QUIET")
                rsi_brk_c = "#22c55e" if rsi_brk <= 60 else "#f59e0b" if rsi_brk <= 70 else "#ef4444"
                em_tgt_c  = {"IGNITING":"#f59e0b","BUILDING":"#22c55e","COILING":"#a78bfa","LATENT":"#38bdf8","QUIET":"#475569"}.get(em_lbl_brk,"#475569")
                _em_friendly_brk = {"IGNITING":"Launching","BUILDING":"Building","COILING":"Tightening","LATENT":"Warming","QUIET":"Dormant"}.get(em_lbl_brk, em_lbl_brk)
                _em_tgt_hint = {"IGNITING":"T1–T3 wide","BUILDING":"T1–T2 solid","COILING":"T1 target","LATENT":"T1 only","QUIET":"tight tgts"}.get(em_lbl_brk,"—")

                cells_html = (
                    _phase_intel_cell("Vol Spike",     f"{rvol_brk:.1f}× avg",                   rv_brk_c, dim=rvol_brk < 1.2)
                    + _phase_intel_cell("Volume OK",   "Confirmed ✓" if vol_conf else "Not yet",  vol_c,   dim=not vol_conf)
                    + _phase_intel_cell("Weekly Trend","Aligned ✓" if htf_up else "Against ✗",   htf_c,   dim=not htf_up)
                    + _phase_intel_cell("Setup Type",   setup_brk[:10] if setup_brk else "—",    stp_c,   dim=not setup_brk)
                    + _phase_intel_cell("RSI",          f"{rsi_brk:.0f}" if rsi_brk else "—",   rsi_brk_c, dim=not rsi_brk)
                    + _phase_intel_cell("Target Size",  f"{_em_friendly_brk} → {_em_tgt_hint}",       em_tgt_c, dim=em_lbl_brk=="QUIET")
                )

            elif group == "STRONG_BUY":
                # ── Strong Buy: Readiness · Confidence · ADX · RS Rank · Breadth · Trend phase ──
                score    = r.get("ReadinessScore", r.get("Score", 0))  # v16.0 FIX-1
                conf     = r.get("Confidence", 0)
                adx      = r.get("ADX", 0)
                rs_rk    = r.get("RS_Rank", 50)
                gated    = r.get("BreadthGated", False)
                phase_sb = r.get("Phase", "")
                trend_up = r.get("TrendUp", True)
                ema_stk  = r.get("EMAStack", False)
                vol_conf_sb = r.get("VolConf", False)
                setup_sb    = r.get("Setup", "—")

                trend_desc = ("Full Stack ↑" if ema_stk else "Above EMA ↑") if trend_up else "Weak ↓"

                adx_c  = "#22c55e" if adx >= 30 else "#f59e0b" if adx >= 20 else "#64748b"
                rs_c   = "#22c55e" if rs_rk >= 70 else "#f59e0b" if rs_rk >= 50 else "#64748b"
                br_c   = "#22c55e" if not gated else "#ef4444"
                td_c   = "#22c55e" if trend_up and ema_stk else "#f59e0b" if trend_up else "#ef4444"
                vc_sb_c= "#22c55e" if vol_conf_sb else "#ef4444"
                stp_sb_c = "#22c55e" if setup_sb in ("breakout","fib") else "#f59e0b"

                cells_html = (
                    _phase_intel_cell("Setup Type",    setup_sb[:10] if setup_sb else "—", stp_sb_c, dim=not setup_sb)
                    + _phase_intel_cell("Vol Confirmed","Yes ✓" if vol_conf_sb else "No ✗",vc_sb_c, dim=not vol_conf_sb)
                    + _phase_intel_cell("Trend Strength",f"{adx:.0f}/100",   adx_c, dim=adx < 18)
                    + _phase_intel_cell("Market Rank", f"Top {100-rs_rk}%",  rs_c,  dim=rs_rk < 40)
                    + _phase_intel_cell("Market Mood", "Healthy ✓" if not gated else "Weak ⚠", br_c, dim=gated)
                    + _phase_intel_cell("Price Phase",  f"{phase_sb} · {trend_desc[:8]}", td_c, dim=not trend_up)
                )

            elif group == "EXTENDED_RISKY":
                # ── Extended/Risky: exhaustion signals + caution metrics ───────
                ext_n   = r.get("ExtN", 0)
                ext_lb  = r.get("ExtLabels", [])
                rsi     = r.get("RSI", 50)
                atr     = r.get("ATR", 0)
                rvol    = r.get("RVOL", 1.0)
                ltp     = r.get("LTP", 0)
                ema200  = r.get("EMA200")   # None when n < 200 — do NOT fall back to ltp
                dist_em = abs(ltp - ema200) / ema200 * 100 if ema200 else 0

                ex_c    = "#ef4444" if ext_n >= 3 else "#f59e0b" if ext_n >= 2 else "#64748b"
                rsi_c   = "#ef4444" if rsi >= 75 else "#f59e0b" if rsi >= 68 else "#64748b"
                rv_c    = "#ef4444" if rvol >= 2.5 else "#f59e0b" if rvol >= 1.8 else "#64748b"
                dist_c  = "#ef4444" if dist_em > 20 else "#f59e0b" if dist_em > 12 else "#64748b"

                ext_tag = " · ".join(ext_lb[:2]) if ext_lb else "—"
                cells_html = (
                    _phase_intel_cell("Warning Count", f"{ext_n} signals",      ex_c,   dim=ext_n < 1)
                    + _phase_intel_cell("Warning Type", ext_tag[:14],            ex_c,   dim=ext_n < 1)
                    + _phase_intel_cell("Overbought",   f"RSI {rsi:.0f}",        rsi_c,  dim=rsi < 65)
                    + _phase_intel_cell("Vol Spike",    f"{rvol:.1f}× avg",      rv_c,   dim=rvol < 1.5)
                    + _phase_intel_cell("Extended %",   f"{dist_em:.1f}% above" if ema200 else "—", dist_c, dim=not ema200 or dist_em < 8)
                    + _phase_intel_cell("Volatility",   f"{atr:.2f}" if atr else "—", "#f59e0b", dim=not atr)
                )

            elif group == "SHORT_SELL":
                # ── Short Sell: bearish pressure + weakness confirmation ───────
                ss_s   = r.get("ShortScore", 0)
                ss_v   = r.get("ShortVerdict", SHORT_SKIP)
                rsi    = r.get("RSI", 50)
                adx    = r.get("ADX", 0)
                rvol   = r.get("RVOL", 1.0)
                mtf_s  = r.get("MTFScore", 50)
                rs_rk  = r.get("RS_Rank", 50)
                sm_v   = r.get("SmartMoneyVerdict", "NEUTRAL")

                ss_c   = "#cc2244" if ss_s >= 65 else "#f59e0b" if ss_s >= 45 else "#64748b"
                rsi_c  = "#22c55e" if rsi <= 40 else "#f59e0b" if rsi <= 50 else "#ef4444"
                rs_c   = "#cc2244" if rs_rk < 30 else "#f59e0b" if rs_rk < 50 else "#64748b"
                sm_c   = "#cc2244" if sm_v == "DISTRIBUTING" else "#64748b"
                mtf_c  = "#cc2244" if mtf_s <= 38 else "#f59e0b" if mtf_s <= 48 else "#64748b"

                cells_html = (
                    _phase_intel_cell("Short Score",   f"{ss_s:.0f}/100",    ss_c,  dim=ss_s < 25)
                    + _phase_intel_cell("Short Signal",  ss_v[:8],            ss_c,  dim=ss_s < 25)
                    + _phase_intel_cell("Momentum Weak", f"RSI {rsi:.0f}",    rsi_c, dim=rsi > 55)
                    + _phase_intel_cell("Market Rank",   f"Bottom {rs_rk}%",  rs_c,  dim=rs_rk >= 50)
                    + _phase_intel_cell("Bear Timeframes",f"{mtf_s:.0f}/100", mtf_c, dim=mtf_s > 50)
                    + _phase_intel_cell("Institutions",  _hl(sm_v)[:14],      sm_c,  dim=sm_v != "DISTRIBUTING")
                )

            return (
                f'<div style="padding:6px 10px 7px;border-top:1px solid #1e2a3a;">'
                f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px;">'
                f'<div style="color:{grp_color};font-size:7.5px;font-weight:700;'
                f'letter-spacing:.1em;text-transform:uppercase;">{grp_label}</div>'
                f'<div style="width:36px;height:2px;border-radius:1px;background:{grp_color}44;">'
                f'<div style="height:2px;border-radius:1px;background:{grp_color};width:100%;"></div></div>'
                f'</div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;">'
                + cells_html
                + f'</div></div>'
            )

        def make_card(i, r, border_color=None, show_entry=True):
            sym   = r["Symbol"]; act = r["Action"]; ltp = r["LTP"]; chg = r["%Change"]
            score = r["Score"];  phase = r.get("Phase", PHASE_IDLE); conf = r.get("Confidence", 0)
            entry = r.get("Entry"); sl = r.get("SL"); t1 = r.get("T1"); t2 = r.get("T2"); t3 = r.get("T3")
            ext_n = r.get("ExtN", 0); ext_labels = r.get("ExtLabels", [])
            sector = r.get("Sector", SECTOR_MAP.get(sym, "—"))
            is_stale = sym in stale_syms

            # v16.0 FIX-1/7: use ReadinessScore as hero, extract TradeIntent
            readiness    = r.get("ReadinessScore", score)
            trade_intent = r.get("TradeIntent", "IGNORE")
            ti_color     = r.get("TradeIntentColor", "#64748b")
            ti_icon      = r.get("TradeIntentIcon", "⚪")
            ti_detail    = r.get("TradeIntentDetail", "")

            # Border colour:
            # • When called from a lifecycle layer, `border_color` is the layer's
            #   identity colour (passed in) — all cards in the same layer look uniform.
            # • Only override with per-stock signal when no layer colour was provided,
            #   OR when ext_n >= 3 (exhaustion — always flag regardless of layer).
            if ext_n >= 3:
                border_color = "#ef4444cc"   # exhaustion always overrides
            elif border_color is None:
                # Fallback: derive from individual stock's TradeIntent
                if trade_intent == "BUY NOW" and act == "STRONG BUY":
                    border_color = "#22c55ecc"
                elif trade_intent == "BUY NOW":
                    border_color = "#22c55e99"
                elif trade_intent == "NOT YET" and act == "PRE-CONFIRM":
                    border_color = "#a78bfa99"
                elif trade_intent == "NOT YET":
                    border_color = "#f59e0b88"
                else:
                    border_color = "#37415166"  # IGNORE — muted gray
            # else: use the layer colour as-is (already set by caller)

            # ── Derived values ────────────────────────────────────────────────
            chg_str = f"+{chg:.2f}%" if chg >= 0 else f"{chg:.2f}%"
            chg_col = "#22c55e" if chg >= 0 else "#ef4444"
            chg_arr = "▲" if chg >= 0 else "▼"
            act_bg, act_brd, act_txt = _action_colors(act)
            phase_col = _phase_color(phase)
            conf_col  = _conf_color(conf)
            phase_icon = {"BREAKOUT":"🚀","CONT":"↗","ENTRY":"⚡","SETUP":"◎","IDLE":"–","EXIT":"↘"}.get(phase,"")
            ph_arrow   = get_phase_arrow(sym)
            score_col  = "#f59e0b" if act == "STRONG BUY" else "#22c55e" if act in ("BUY","PRE-CONFIRM") else "#3b82f6"

            def _p(v):
                if v is None: return "—"
                try: return f"₹{int(round(v)):,}"
                except: return "—"

            ref = entry if (show_entry and entry and entry != ltp) else ltp
            risk_pct = reward_pct = rr = None
            if ref and sl:
                risk = ref - sl
                if risk > 0:
                    risk_pct = risk / ref * 100
                    tgt = t1  # T1 is the first real target; R:R based on that
                    if tgt:
                        reward_pct = (tgt - ref) / ref * 100
                        rr = reward_pct / risk_pct
            rr_col = "#22c55e" if (rr and rr >= 2) else "#f59e0b" if (rr and rr >= 1.5) else "#64748b"
            rr_str = f"{rr:.1f}×" if rr else "—"
            entry_disp = f"₹{entry:,.0f}" if (show_entry and entry and entry != ltp) else f"₹{ltp:,.2f}"

            stale_dot = ' <span style="color:#475569;font-size:9px;">⏱</span>' if is_stale else ""
            ed = st.session_state.get("earnings_map", {}).get(sym)
            earn_html = (f'<span style="background:#7f1d1d;border:1px solid #ef4444;color:#fca5a5;'
                         f'padding:1px 5px;border-radius:3px;font-size:9px;font-weight:700;margin-left:4px;">'
                         f'⚠ RESULTS {ed}</span>') if ed else ""

            # ── Why Now signals ───────────────────────────────────────────────
            sigs = _unique_signals(r)
            sig_rows = "".join(
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'padding:4px 0;border-bottom:1px solid #12182a;">'
                f'<span style="color:#94a3b8;font-family:JetBrains Mono,monospace;font-size:9px;'
                f'width:90px;flex-shrink:0;">{s["label"]}</span>'
                f'<div style="flex:1;margin:0 6px;background:#12182a;border-radius:2px;height:3px;">'
                f'<div style="background:{s["color"]};width:{min(s["rank"],100)}%;height:3px;border-radius:2px;"></div></div>'
                f'<span style="color:{s["color"]};font-family:JetBrains Mono,monospace;font-size:9px;'
                f'font-weight:600;text-align:right;min-width:70px;">{s["value"]}</span>'
                f'</div>'
                for s in sigs
            ) if sigs else '<div style="color:#334155;font-size:9px;padding:4px 0;">No dominant signals</div>'

            # ── Phase-aware intelligence grid (v15.7+ upgrade) ───────────────
            # Shows ONLY the indicators relevant to the next decision for this stock's lifecycle stage.
            ui_group  = _resolve_ui_group(r)
            intel_grid = _build_phase_intel(r, ui_group)

            caution = _caution_line(r, group=ui_group)
            caution_html = (
                f'<div style="display:flex;gap:4px;align-items:flex-start;'
                f'padding:4px 6px;margin-top:4px;background:#1c0700;border-left:2px solid #f59e0b;border-radius:2px;">'
                f'<span style="color:#f59e0b;font-size:9px;flex-shrink:0;">⚠</span>'
                f'<span style="color:#fbbf24;font-size:9px;">{caution}</span></div>'
            ) if caution else ""

            # ── Exhaustion warning ────────────────────────────────────────────
            ext_html = ""
            if ext_n > 0:
                ec = "#fca5a5" if ext_n >= 3 else "#fbbf24"
                eb = "#3b1a0a" if ext_n >= 3 else "#2a1e00"
                skip_warn = " — SKIP ENTRY" if ext_n >= 3 else " — reduce size"
                pills = "  ".join(f'⚠ {lb}' for lb in ext_labels[:2])
                ext_html = (f'<div style="padding:4px 12px;background:{eb};border-top:1px solid #1e293b;">'
                            f'<span style="color:{ec};font-size:9px;">{pills}{skip_warn}</span></div>')

            # ── R:R visual bar ────────────────────────────────────────────────
            rr_bar_pct = min(int((rr / 4) * 100), 100) if rr else 0

            # ── Compact metrics strip (group-aware to avoid duplicating intel grid) ──
            c_rsi_val = r.get("RSI", "—")
            c_rs_rank = r.get("RS_Rank", 50)
            c_atr_val = r.get("ATR", "—")
            c_adx_val = r.get("ADX", "—")
            # Compact metrics strip (group-aware; RSI always shown — CONF% was duplicate of intel grid)
            conf_metrics_grid = (
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;'
                f'gap:0;padding:4px 10px;border-top:1px solid #1e2a3a;">'
                f'<div><div style="color:#475569;font-size:7.5px;">RSI</div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:#94a3b8;font-size:10px;font-weight:600;">{c_rsi_val}</div></div>'
                f'<div><div style="color:#475569;font-size:7.5px;">RS RNK</div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:#94a3b8;font-size:10px;font-weight:600;">{c_rs_rank}</div></div>'
                f'<div><div style="color:#475569;font-size:7.5px;">ATR</div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:#94a3b8;font-size:10px;font-weight:600;">{c_atr_val}</div></div>'
                f'<div><div style="color:#475569;font-size:7.5px;">ADX</div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:#94a3b8;font-size:10px;font-weight:600;">{c_adx_val}</div></div>'
                f'</div>'
            )

            # ── ReadinessBand vars (computed outside the return f-string) ─────
            _rband     = r.get("ReadinessBand", "WEAK")
            _rband_col = r.get("BandColor", "#475569")
            _rband_note= r.get("BandNote", "Watch only")
            _rcause    = r.get("CauseScore", readiness)
            _rtiming   = r.get("TimingScore", readiness)
            _rctx      = r.get("ContextScore", readiness)
            _rop       = r.get("OrthoPenalty", 0.0)
            _ortho_html= (f'<span style="color:#ef4444;font-size:6px;"> -{_rop:.0f}✗</span>'
                          if _rop > 0 else '')
            _price_band_block = (
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'padding:6px 12px;border-bottom:1px solid #1e2a3a;">'
                f'<div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:#f8fafc;font-size:18px;'
                f'font-weight:700;line-height:1;">₹{ltp:,.2f}</div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:{chg_col};font-size:9px;'
                f'margin-top:2px;">{chg_arr} {chg_str}</div>'
                f'</div>'
                + _em_stage_strip(r.get("EmLabel", "QUIET"))
                + f'</div>'
            )

            return (
                f'<div style="background:#07101e;border:1px solid {border_color};'
                f'border-top:3px solid {border_color[:7]};border-radius:10px;'
                f'overflow:hidden;min-width:210px;max-width:300px;flex:1 1 210px;">'

                # ── Header: symbol · phase · action ──────────────────────────
                f'<div style="display:flex;align-items:center;padding:9px 12px 8px;'
                f'gap:7px;background:#0b1422;border-bottom:1px solid #1e2a3a;">'
                f'<div style="flex:1;min-width:0;">'
                f'<div style="display:flex;align-items:center;gap:5px;">'
                f'<span style="font-family:Syne,sans-serif;color:#f1f5f9;font-size:15px;font-weight:700;">{sym}</span>'
                f'{stale_dot}{earn_html}</div>'
                f'<div style="display:flex;align-items:center;gap:4px;margin-top:3px;">'
                f'<span style="background:{phase_col}20;color:{phase_col};font-size:9px;font-weight:600;'
                f'padding:1px 6px;border-radius:3px;border:1px solid {phase_col}40;">'
                f'{phase_icon} {phase}{(" "+ph_arrow) if ph_arrow else ""}</span>'
                f'<span style="color:#475569;font-size:9px;">{sector}</span>'
                f'</div></div>'
                f'<span style="background:{act_bg};border:1px solid {act_brd};color:{act_txt};'
                f'padding:3px 9px;border-radius:5px;font-size:10px;font-weight:700;">{act}</span>'
                f'</div>'

                # ── v16.0 FIX-7: TradeIntent banner — dominant plain-English decision ──
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'padding:5px 12px;background:{ti_color}15;border-bottom:1px solid {ti_color}30;">'
                f'<div style="display:flex;align-items:center;gap:6px;">'
                f'<span style="font-size:13px;">{ti_icon}</span>'
                f'<span style="font-family:Syne,sans-serif;color:{ti_color};font-size:12px;'
                f'font-weight:700;letter-spacing:.04em;">{trade_intent}</span>'
                f'</div>'
                f'<span style="color:{ti_color}99;font-size:8.5px;">{ti_detail}</span>'
                f'</div>'

                # ── Price + ReadinessBand (false precision removed) ────────────
                + _price_band_block

                # ── Compact trade strip: entry · SL · T1 · R:R · T2 T3 ────────
                + f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;'
                f'gap:1px;background:#1e2a3a;border-bottom:1px solid #1e2a3a;">'
                f'<div style="background:#050d18;padding:5px 8px;">'
                f'<div style="color:#475569;font-size:7.5px;">ENTRY</div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:#e2e8f0;font-size:10px;font-weight:700;">{entry_disp}</div>'
                f'</div>'
                f'<div style="background:#0e0808;padding:5px 8px;">'
                f'<div style="color:#475569;font-size:7.5px;">SL</div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:#f87171;font-size:10px;font-weight:700;">{_p(sl)}</div>'
                f'</div>'
                f'<div style="background:#07120a;padding:5px 8px;">'
                f'<div style="color:#475569;font-size:7.5px;">T1</div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:#4ade80;font-size:10px;font-weight:700;">{_p(t1)}</div>'
                f'</div>'
                f'<div style="background:#050d18;padding:5px 8px;">'
                f'<div style="color:#475569;font-size:7.5px;">R:R</div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:{rr_col};font-size:10px;font-weight:700;">{rr_str}</div>'
                f'</div>'
                f'</div>'
                # T2 / T3 as small chips if present
                + (f'<div style="display:flex;gap:6px;padding:4px 10px;border-bottom:1px solid #1e2a3a;background:#050d18;">'
                   f'<span style="color:#334155;font-size:8px;align-self:center;">ext →</span>'
                   + (f'<span style="background:#07120a;border:1px solid #14532d44;color:#86efac;font-family:JetBrains Mono,monospace;font-size:9px;padding:1px 6px;border-radius:3px;">T2 {_p(t2)}</span>' if t2 else '')
                   + (f'<span style="background:#07120a;border:1px solid #14532d33;color:#4ade80;font-family:JetBrains Mono,monospace;font-size:9px;padding:1px 6px;border-radius:3px;">T3 {_p(t3)}</span>' if t3 else '')
                   + f'<div style="flex:1;background:#1e2a3a;border-radius:2px;height:2px;align-self:center;margin-left:4px;">'
                   f'<div style="background:{rr_col};width:{min(int((rr/3)*100),100) if rr else 0}%;height:2px;border-radius:2px;"></div></div>'
                   f'</div>' if (t2 or t3) else '')

                # ── Caution alert ─────────────────────────────────────────────
                + (f'<div style="padding:3px 12px 2px;border-top:1px solid #1e2a3a;">'
                   + caution_html
                   + f'</div>' if caution else '')

                # ── Compact metrics strip (RSI · RS RNK · ATR · ADX) ──────────
                + conf_metrics_grid

                # ── Phase-aware intelligence (ONLY relevant signals) ──────────
                + intel_grid

                # ── Exhaustion warning (prominent at bottom) ──────────────────
                + ext_html +

                f'</div>'
            )

        ACTIONABLE_PHASES={PHASE_ENTRY,PHASE_CONT,PHASE_BRK}
        actionable=[r for r in st.session_state.get("results", [])
                    if r.get("Phase") in ACTIONABLE_PHASES and r["Action"] in ("BUY","STRONG BUY","PRE-CONFIRM")]
        phase_rank={PHASE_BRK:0,PHASE_CONT:1,PHASE_ENTRY:2}
        actionable.sort(key=lambda x:(phase_rank.get(x.get("Phase"),9),-x["Score"]))
        top_act=actionable[:15]

        # ── v15.5: EMERGING MOMENTUM CARD ─────────────────────────────────────
        _EM_COLORS = {
            "IGNITING": ("#f59e0b","#f59e0b22"),
            "BUILDING": ("#22c55e","#22c55e22"),
            "COILING":  ("#8b5cf6","#8b5cf622"),
            "LATENT":   ("#38bdf8","#38bdf822"),
            "QUIET":    ("#475569","#47556922"),
        }
        _EM_COMPONENTS = [
            ("RS Accel",    "EmRSAccel",     15, "📈"),
            ("ATR Cmprss",  "EmATRCompress", 15, "🗜"),
            ("RVOL Accel",  "EmRVolAccel",   15, "📊"),
            ("EMA Conv",    "EmEMAConv",     15, "🔀"),
            ("Sqz Press",   "EmSqzPressure", 15, "🔄"),
            ("Sector Mom",  "EmSectorMom",   10, "🏭"),
            ("Range Exp",   "EmORExpansion", 15, "🚀"),
        ]

        # ── Plain-English EmLabel names + 4-step visual strip ─────────────────
        # EmLabel internal → user-facing friendly name
        _EM_FRIENDLY = {
            "QUIET":    "Dormant",
            "LATENT":   "Warming",      # Energy just starting to stir
            "COILING":  "Tightening",   # Spring wound tight, range contracting
            "BUILDING": "Building",     # Momentum actively building
            "IGNITING": "Launching",    # About to break out
        }
        # 4 visible stages (low → high); QUIET is hidden
        _EM_STAGE_ORDER  = ["WARMING", "TIGHTENING", "BUILDING", "LAUNCHING"]
        _EM_STAGE_FROM   = {  # internal EmLabel → stage strip label
            "LATENT":   "WARMING",
            "COILING":  "TIGHTENING",
            "BUILDING": "BUILDING",
            "IGNITING": "LAUNCHING",
        }
        _EM_STAGE_COLORS = {
            "WARMING":    "#38bdf8",   # blue   – cool, early
            "TIGHTENING": "#a78bfa",   # violet – coiled, tightening
            "BUILDING":   "#22c55e",   # green  – active build
            "LAUNCHING":  "#f59e0b",   # amber  – imminent breakout
        }

        def _em_stage_strip(em_label):
            """4-step vertical progression strip; highlights current stage. Placed right of LTP."""
            stages  = _EM_STAGE_ORDER
            current = _EM_STAGE_FROM.get(em_label)
            cur_idx = stages.index(current) if current in stages else -1
            pills   = []
            for idx, stage in enumerate(stages):
                col    = _EM_STAGE_COLORS[stage]
                is_cur = (idx == cur_idx)
                is_past= (idx < cur_idx)
                if is_cur:
                    bg, brd, tc, fw, sfx = f"{col}28", col, col, "700", "▲"
                elif is_past:
                    bg, brd, tc, fw, sfx = f"{col}12", f"{col}55", f"{col}77", "500", "✓"
                else:
                    bg, brd, tc, fw, sfx = "#0e1624", "#1e2a3a", "#37415166", "400", ""
                pills.append(
                    f'<div style="padding:2px 5px;border-radius:3px;'
                    f'background:{bg};border:1px solid {brd};margin-bottom:2px;">'
                    f'<div style="color:{tc};font-size:6px;font-weight:{fw};'
                    f'letter-spacing:.01em;white-space:nowrap;text-align:right;">'
                    f'{sfx} {stage}</div>'
                    f'</div>'
                )
            # Reverse so LAUNCHING is at top
            pills = pills[::-1]
            return (
                f'<div style="display:flex;flex-direction:column;justify-content:center;min-width:72px;">'
                + "".join(pills)
                + f'</div>'
            )

        # ── Layer R:R colour palette ──────────────────────────────────────────────
        # Each layer has a single identity colour that reflects its R:R profile
        # relative to the overall lifecycle, NOT the individual stock's signal.
        #   ACCUMULATING    → cyan/teal  (#38bdf8)  — watch / early-stage, low immediate R:R
        #   EMERGING_MOMENTUM→ violet    (#a78bfa)  — building / pre-confirm
        #   BREAKOUT_READY  → amber      (#f59e0b)  — near-actionable
        #   STRONG_BUY      → green      (#22c55e)  — act now, best R:R
        #   EXTENDED_RISKY  → red        (#ef4444)  — risky, degraded R:R
        _LAYER_COLORS = {
            "ACCUMULATING":      "#38bdf8",   # cyan  — watch, building base
            "EMERGING_MOMENTUM": "#a78bfa",   # violet — coiling, pre-confirm
            "BREAKOUT_READY":    "#f59e0b",   # amber — near-entry
            "STRONG_BUY":        "#22c55e",   # green — act now
            "EXTENDED_RISKY":    "#ef4444",   # red   — avoid / reduce
        }

        def make_emerging_card(i, r, layer_color=None):
            sym   = r["Symbol"]; ltp = r["LTP"]; chg = r["%Change"]
            em    = r.get("EmScore", 0); lbl = r.get("EmLabel","QUIET")
            pca   = r.get("PCAScore", 0); pca_lbl = r.get("PCALabel","NONE")
            phase = r.get("Phase", PHASE_IDLE)
            act   = r.get("Action","SKIP"); sector = r.get("Sector","—")
            # Combined readiness: EmScore × 0.55 + PCA × 0.45
            readiness = round(em * 0.55 + pca * 0.45, 1)
            # TradeIntent still drives the BADGE colour inside the card,
            # but the BORDER is now the layer colour (passed in by the caller).
            _ti        = r.get("TradeIntent", "IGNORE")
            _ti_color  = r.get("TradeIntentColor", "#64748b")
            # em_c = layer colour if provided, else fall back to TradeIntent colour
            em_c  = layer_color if layer_color else _ti_color
            em_bg = f"{em_c}18"   # header tint uses layer colour
            # Border: uniform layer colour — thickness encodes R:R tier
            #   2px solid   = high-conviction layers (STRONG_BUY, EXTENDED_RISKY)
            #   1.5px solid = watch / building layers (ACCUMULATING, EMERGING_MOMENTUM)
            #   1px solid   = near-entry (BREAKOUT_READY)
            if layer_color in (_LAYER_COLORS["STRONG_BUY"], _LAYER_COLORS["EXTENDED_RISKY"]):
                _border_style = f"2px solid {em_c}cc"
            elif layer_color == _LAYER_COLORS["BREAKOUT_READY"]:
                _border_style = f"2px solid {em_c}aa"
            else:
                _border_style = f"1.5px solid {em_c}99"
            _PCA_COLORS = {
                "ACCUMULATING": ("#22c55e","#22c55e22"),
                "BUILDING":     ("#38bdf8","#38bdf822"),
                "FORMING":      ("#a78bfa","#a78bfa22"),
                "WEAK":         ("#64748b","#64748b22"),
                "NONE":         ("#374151","#37415122"),
            }
            pca_c, pca_bg = _PCA_COLORS.get(pca_lbl, ("#374151","#37415122"))
            chg_col = "#22c55e" if chg >= 0 else "#ef4444"
            chg_arr = "▲" if chg >= 0 else "▼"
            chg_str = f"+{chg:.2f}%" if chg >= 0 else f"{chg:.2f}%"
            phase_col = _phase_color(phase)
            act_bg, act_brd, act_txt = _action_colors(act)

            # ── Squeeze / Vol-Contraction strip (price-area only) ─────────────
            sqz_badge = ('<span style="background:#8b5cf622;border:1px solid #8b5cf655;color:#a78bfa;'
                         'padding:1px 5px;border-radius:3px;font-size:9px;margin-right:3px;">🔄 SQZ</span>'
                         if r.get("Squeeze") else "")
            vc_badge  = ('<span style="background:#0ea5e922;border:1px solid #0ea5e955;color:#38bdf8;'
                         'padding:1px 5px;border-radius:3px;font-size:9px;margin-right:3px;">VC</span>'
                         if r.get("VolRatio", 1.0) < 0.75 else "")

            # ── Phase-aware intelligence grid (v15.7+ upgrade) ───────────────
            # Emerging cards always use EMERGING_MOMENTUM group unless the stock
            # has already moved into a more advanced lifecycle stage.
            em_ui_group = _resolve_ui_group(r)
            # Override to EMERGING_MOMENTUM if the stock is pre-breakout coiling
            if em_ui_group not in ("EXTENDED_RISKY", "SHORT_SELL", "STRONG_BUY", "BREAKOUT_READY", "ACCUMULATING"):
                em_ui_group = "EMERGING_MOMENTUM"
            intel_grid = _build_phase_intel(r, em_ui_group)

            # ── Compact metrics grid — group-aware (hide ADX/ATR for Accum/Emerging) ──
            rsi_val  = r.get("RSI","—"); rs_rank = r.get("RS_Rank", 50)
            atr_val  = r.get("ATR","—"); adx_val = r.get("ADX","—")
            # v16.0 FIX-1/3/4: use ReadinessScore + PCAPercentile instead of raw SM + PCA scores
            ready_val  = r.get("ReadinessScore", r.get("Score", 0))
            pca_pct_v  = r.get("PCAPercentile", 0)
            em_score_v = r.get("EmScore", 0)
            # Pre-compute friendly label once (used in header chip, metrics, footer)
            _em_lbl_friendly = {"IGNITING":"Launching","BUILDING":"Building","COILING":"Tightening","LATENT":"Warming","QUIET":"Dormant"}.get(lbl, lbl)
            _em_lbl_col_card  = {"IGNITING":"#f59e0b","BUILDING":"#22c55e","COILING":"#a78bfa","LATENT":"#38bdf8","QUIET":"#475569"}.get(lbl,"#475569")
            # TradeIntent for the emerging card banner
            em_ti_color  = r.get("TradeIntentColor", "#64748b")
            em_ti_icon   = r.get("TradeIntentIcon", "⚪")
            em_ti        = r.get("TradeIntent", "IGNORE")
            em_ti_detail = r.get("TradeIntentDetail", "")
            if em_ui_group == "ACCUMULATING":
                # ACCUMULATING: show Readiness, PCA %, RS Rank, AccumStage
                acc_stg = r.get("AccumStage", "NONE")
                metrics_grid = (
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;'
                    f'gap:0;padding:6px 12px;border-top:1px solid #1e1e40;">'
                    f'<div><div style="color:#475569;font-size:8px;">READY SCR</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;color:#22c55e;font-size:11px;font-weight:600;">{ready_val:.0f}</div></div>'
                    f'<div><div style="color:#475569;font-size:8px;">BUY PRESS %</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;color:#a78bfa;font-size:11px;font-weight:600;">Top {100-pca_pct_v}%</div></div>'
                    f'<div><div style="color:#475569;font-size:8px;">RS RNK</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;color:#94a3b8;font-size:11px;font-weight:600;">{rs_rank}</div></div>'
                    f'<div><div style="color:#475569;font-size:8px;">BASE STG</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;color:#f59e0b;font-size:11px;font-weight:600;">{acc_stg}</div></div>'
                    f'</div>'
                )
            elif em_ui_group == "EMERGING_MOMENTUM":
                # EMERGING: Coil State removed (now in stage strip); show Readiness, RSI, RS Rank, Sector Mom
                sec_m_val = r.get("EmSectorMom", 0)
                metrics_grid = (
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;'
                    f'gap:0;padding:6px 12px;border-top:1px solid #1e1e40;">'
                    f'<div><div style="color:#475569;font-size:7.5px;">READY SCR</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;color:{_em_lbl_col_card};font-size:11px;font-weight:700;">{em_score_v:.0f}</div></div>'
                    f'<div><div style="color:#475569;font-size:7.5px;">RSI</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;color:#94a3b8;font-size:11px;font-weight:600;">{rsi_val}</div></div>'
                    f'<div><div style="color:#475569;font-size:7.5px;">RS RNK</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;color:#94a3b8;font-size:11px;font-weight:600;">{rs_rank}</div></div>'
                    f'<div><div style="color:#475569;font-size:7.5px;">SEC MOM</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;color:#22c55e;font-size:11px;font-weight:600;">{sec_m_val:.0f}</div></div>'
                    f'</div>'
                )
            else:
                metrics_grid = (
                    f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;'
                    f'gap:0;padding:6px 12px;border-top:1px solid #1e1e40;">'
                    f'<div><div style="color:#475569;font-size:8px;">RSI</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;color:#94a3b8;font-size:11px;font-weight:600;">{rsi_val}</div></div>'
                    f'<div><div style="color:#475569;font-size:8px;">RS RNK</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;color:#94a3b8;font-size:11px;font-weight:600;">{rs_rank}</div></div>'
                    f'<div><div style="color:#475569;font-size:8px;">ATR</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;color:#94a3b8;font-size:11px;font-weight:600;">{atr_val}</div></div>'
                    f'<div><div style="color:#475569;font-size:8px;">ADX</div>'
                    f'<div style="font-family:JetBrains Mono,monospace;color:#94a3b8;font-size:11px;font-weight:600;">{adx_val}</div></div>'
                    f'</div>'
                )

            return (
                f'<div style="background:#0e0e1c;border:{_border_style};border-radius:12px;'
                f'overflow:hidden;min-width:210px;max-width:300px;flex:1 1 210px;">'
                # ── Header: symbol + single Readiness chip + phase/action ──────────
                # v16.0 FIX-1/4: replaced dual EM+PCA chips with one ReadinessScore
                f'<div style="background:{em_bg};border-bottom:1px solid {em_c}33;padding:8px 12px 7px;'
                f'display:flex;align-items:center;gap:8px;">'
                f'<div style="flex:1;">'
                f'<div style="font-family:Syne,sans-serif;color:#e8e8f4;font-size:15px;font-weight:700;">{sym}</div>'
                f'<div style="font-size:9px;color:#94a3b8;">{sector}</div>'
                f'</div>'
                f'<div style="text-align:right;">'
                f'<div style="display:flex;gap:3px;justify-content:flex-end;flex-wrap:wrap;">'
                f'<span style="background:{em_c};color:#0a0a0f;font-family:JetBrains Mono,monospace;'
                f'font-size:11px;font-weight:700;padding:3px 8px;border-radius:4px;">'
                f'{_em_lbl_friendly} {readiness:.0f}</span>'
                f'</div>'
                f'<div style="margin-top:4px;display:flex;gap:3px;justify-content:flex-end;">'
                f'<span style="background:{phase_col}22;border:1px solid {phase_col}55;color:{phase_col};'
                f'padding:1px 5px;border-radius:3px;font-size:9px;">{phase}</span>'
                f'<span style="background:{act_bg};border:1px solid {act_brd};color:{act_txt};'
                f'padding:1px 5px;border-radius:3px;font-size:9px;font-weight:600;">{act}</span>'
                f'</div></div></div>'
                # ── Price row + SQZ/VC strip + vertical stage strip ───────────────
                f'<div style="padding:8px 12px 5px;display:flex;justify-content:space-between;align-items:center;">'
                f'<div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:#e8e8f4;font-size:18px;font-weight:600;">₹{ltp:,.2f}</div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:{chg_col};font-size:10px;">{chg_arr} {chg_str}</div>'
                f'<div style="margin-top:3px;">{sqz_badge}{vc_badge}</div>'
                f'</div>'
                + _em_stage_strip(lbl)
                + f'</div>'
                # ── v16.0 FIX-7: TradeIntent banner on emerging cards ─────────────
                + f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'padding:4px 12px;background:{em_ti_color}15;border-top:1px solid {em_ti_color}30;">'
                f'<div style="display:flex;align-items:center;gap:5px;">'
                f'<span style="font-size:11px;">{em_ti_icon}</span>'
                f'<span style="font-family:Syne,sans-serif;color:{em_ti_color};font-size:11px;font-weight:700;">{em_ti}</span>'
                f'</div>'
                f'<span style="color:{em_ti_color}99;font-size:8px;">{em_ti_detail}</span>'
                f'</div>'
                # ── Compact metrics grid ───────────────────────────────────────
                + metrics_grid
                # ── 3×2 Intelligence grid ─────────────────────────────────────
                + intel_grid +
                # ── Footer: SL only ────────────────────────────────────────────
                f'<div style="background:#07070f;border-top:1px solid #1e1e40;padding:5px 12px;">'
                f'<div style="color:#334155;font-size:7.5px;text-transform:uppercase;letter-spacing:.05em;">If it breaks down, exit at</div>'
                f'<div style="color:#f87171;font-family:JetBrains Mono,monospace;font-size:10px;font-weight:600;">SL {"₹{:,.0f}".format(r["SL"]) if r.get("SL") else "—"}</div>'
                f'</div>'
                f'</div>'
            )

        # ══════════════════════════════════════════════════════════════════════
        # 6-LAYER LIFECYCLE UI  (from design inputs)
        # Order: Accumulating → Emerging Momentum → Breakout Ready →
        #        Strong Buy → Extended/Risky → Short Sell
        # Each layer shows ONLY the indicators relevant to the next decision.
        # ══════════════════════════════════════════════════════════════════════

        # ── Bucket all results by lifecycle group ──────────────────────────────
        _all_res = st.session_state.get("results", [])
        _buckets = {g: [] for g in ("ACCUMULATING","EMERGING_MOMENTUM",
                                    "BREAKOUT_READY","STRONG_BUY","EXTENDED_RISKY")}
        for _r in _all_res:
            _g = _resolve_ui_group(_r)
            if _g in _buckets:
                _buckets[_g].append(_r)

        # Sort each bucket by phase-specific priority signals
        _buckets["ACCUMULATING"].sort(
            key=lambda x: (x.get("ReadinessScore", x.get("Score",0)), x.get("AccumSequenceScore",0)),
            reverse=True
        )
        _buckets["EMERGING_MOMENTUM"].sort(
            key=lambda x: (
                1 if x.get("Action") == "PRE-CONFIRM" else 0,
                x.get("ReadinessScore", x.get("EmScore",0)),
                x.get("EmRSAccel",0),
                x.get("EmSectorMom",0)
            ),
            reverse=True
        )
        _buckets["BREAKOUT_READY"].sort(
            key=lambda x: (
                x.get("Confidence",0),
                1 if x.get("VolConf") else 0,
                -(abs((x.get("Entry") or x.get("LTP",0)) - x.get("LTP",0)) / max(x.get("LTP",1),1) * 100)
            ),
            reverse=True
        )
        _buckets["STRONG_BUY"].sort(
            key=lambda x: (x.get("ReadinessScore", x.get("Score",0)), x.get("ADX",0), x.get("RS_Rank",0)),
            reverse=True
        )
        _buckets["EXTENDED_RISKY"].sort(key=lambda x: x.get("Score",0), reverse=True)

        # ══════════════════════════════════════════════════════════════════════
        # TODAY'S 5 — Auto-filtered shortlist for direct action
        # Filter: StructureScore ≥ 55 AND TimingScore ≥ 50 AND
        #         Action in (BUY, STRONG BUY) AND ExtN == 0
        # Rank:   StructureScore × TimingScore (joint confidence, not just one axis)
        # Shows 5 names max with one plain-English reason each.
        # ══════════════════════════════════════════════════════════════════════
        def _today5_reason(r):
            """One sentence explaining why this stock made the list."""
            em   = r.get("EmLabel", "QUIET")
            smv  = r.get("SmartMoneyVerdict", "NEUTRAL")
            ast  = r.get("AccumStage", "NONE")
            sqz  = r.get("Squeeze", False)
            sqzb = r.get("SqzBars", 0)
            adx  = r.get("ADX", 0)
            htf  = r.get("HTFUp", True)
            conf = r.get("Confidence", 0)
            rsi  = r.get("RSI", 50)
            volc = r.get("VolConf", False)
            setup= r.get("Setup", "")
            pca  = r.get("PCAScore", 0)
            rs   = r.get("RS_Rank", 50)
            micro= r.get("MicroScore", 50)
            sc   = r.get("StructureScore", 0)
            tc   = r.get("TimingScore", 0)

            # Pick the single most differentiating reason
            if smv in ("MARKUP_READY", "ACCUMULATING") and em in ("IGNITING","BUILDING"):
                return f"Institutions in {smv.lower().replace('_',' ')} phase + momentum {em.lower()} — rare alignment"
            if sqz and sqzb >= 10:
                return f"Price coiled {sqzb} bars in squeeze with {em.lower()} energy — imminent release"
            if smv == "MARKUP_READY" and htf:
                return f"Smart money signalling markup ready, weekly trend confirmed"
            if em == "IGNITING" and volc:
                return f"Momentum igniting with volume confirmation — energy releasing now"
            if ast in ("1C","2A") and sc >= 65:
                _ast_desc = "ready to break out" if ast == "1C" else "early uptrend"
                return f"Base stage {ast} ({_ast_desc}) with strong structure score {sc:.0f}"
            if adx >= 30 and rs >= 70 and htf:
                return f"Strong trend (ADX {adx:.0f}) + top {100-rs}% RS rank + weekly aligned"
            if pca >= 65 and micro >= 65:
                return f"Institutional accumulation (PCA {pca:.0f}) confirmed by intrabar order flow"
            if setup == "breakout" and volc and conf >= 70:
                return f"Breakout setup, volume confirmed, {conf}% confidence"
            if tc >= 70:
                return f"Timing score {tc:.0f}/100 — multiple leading indicators aligned simultaneously"
            return f"Structure {sc:.0f} · Timing {tc:.0f} — both axes above threshold with no exhaustion"
       

        st.markdown('<div style="margin-bottom:8px;"></div>', unsafe_allow_html=True)

        # ── LAYER GUIDE — what each layer means and what to do ────────────────
        with st.expander("📖 Layer Guide — what each section means and when to act", expanded=False):
            st.markdown("""
<div style="font-family:JetBrains Mono,monospace;font-size:10px;color:#94a3b8;line-height:1.9;">

<div style="color:#38bdf8;font-weight:700;font-size:11px;margin-bottom:2px;">🛡 ACCUMULATING</div>
<div style="color:#64748b;margin-bottom:10px;padding-left:12px;">
<b>What it means:</b> Institutional money is quietly building a position. Price is flat or rangebound — that's intentional, not weakness.<br>
<b>What the engine measures:</b> PCA (CMF trend, hidden accumulation, effort/result, vol asymmetry) + SmartMoney verdict + Wyckoff base stage.<br>
<b>Key numbers:</b> StructureScore tells you how strong the accumulation is. Days Building tells you how mature the base is.<br>
<b>What to do:</b> <span style="color:#f59e0b;">Watch, do not buy.</span> Wait for it to graduate to Emerging (TimingScore rising) or Breakout Ready (phase ENTRY/BREAKOUT).<br>
<b>Upgrade trigger:</b> EmLabel changes from QUIET → COILING → BUILDING + AccumStage moves to 1C or 2A.
</div>

<div style="color:#a78bfa;font-weight:700;font-size:11px;margin-bottom:2px;">🌱 EMERGING MOMENTUM</div>
<div style="color:#64748b;margin-bottom:10px;padding-left:12px;">
<b>What it means:</b> The coil is tightening. ATR is compressing, EMAs are converging, volume is quietly building. Energy is storing before a release.<br>
<b>What the engine measures:</b> EmScore (ATR compression, BB/KC squeeze bars, EMA convergence, RVOL acceleration, RS acceleration).<br>
<b>Key numbers:</b> TimingScore is the signal here. Squeeze ON + COILING or BUILDING = energy stored. PRE-CONFIRM badge = early entry candidate.<br>
<b>What to do:</b> <span style="color:#f59e0b;">Alert only — not an entry yet.</span> Set price alerts. PRE-CONFIRM stocks can be sized at 25–50% of full position as a pre-breakout probe.<br>
<b>Upgrade trigger:</b> Phase moves to ENTRY or BREAKOUT + VolConf = true.
</div>

<div style="color:#f59e0b;font-weight:700;font-size:11px;margin-bottom:2px;">⚡ BREAKOUT READY</div>
<div style="color:#64748b;margin-bottom:10px;padding-left:12px;">
<b>What it means:</b> Price is at or near the trigger level. The setup is valid and entry conditions are either met or one candle away.<br>
<b>What the engine measures:</b> Phase (ENTRY/BREAKOUT), entry proximity, VolConf, HTF alignment, Confidence score, setup type (breakout/fib/squeeze).<br>
<b>Key numbers:</b> RSI (should be 50–70, not overbought), Vol Spike (≥1.5× confirms), Weekly Trend (must be aligned), Confidence (≥60 is actionable).<br>
<b>What to do:</b> <span style="color:#22c55e;">Entry eligible.</span> Buy at the Entry price shown with the SL shown. T1 is the first partial exit. Target size cell shows whether EM energy supports a full run or just T1.<br>
<b>Red flag:</b> Vol Spike dimmed + Weekly Trend showing Against = wait for next bar.
</div>

<div style="color:#22c55e;font-weight:700;font-size:11px;margin-bottom:2px;">🚀 STRONG BUY</div>
<div style="color:#64748b;margin-bottom:10px;padding-left:12px;">
<b>What it means:</b> Price has confirmed and trend is running. This is a momentum continuation play, not an early entry.<br>
<b>What the engine measures:</b> Action = STRONG BUY, Phase = ENTRY/CONT/BREAKOUT, ADX ≥ 25 (trend has strength), RS Rank ≥ 60.<br>
<b>Key numbers:</b> ADX (trend strength — below 20 means the move lacks conviction), RS Rank (top 30% = the stock is leading the market), Breadth (market mood must be healthy).<br>
<b>What to do:</b> <span style="color:#22c55e;">Enter on pullbacks to EMA or with volume confirmation on breakout bars.</span> Use T1 as first target, trail with EMA on the primary TF.<br>
<b>Risk:</b> S:T split matters here — high TimingScore but low StructureScore = momentum without accumulation = fragile. Size down.
</div>

<div style="color:#ef4444;font-weight:700;font-size:11px;margin-bottom:2px;">⚠ EXTENDED / RISKY</div>
<div style="color:#64748b;margin-bottom:10px;padding-left:12px;">
<b>What it means:</b> The move has already happened. Price is stretched from its EMAs, RSI is elevated, or volume is climactic.<br>
<b>What the engine measures:</b> ExtN ≥ 2 (exhaustion flags), RSI ≥ 70, distance from EMA200, RVOL spike (climax volume pattern).<br>
<b>Key numbers:</b> Warning Count (2 = reduce size; 3+ = no entry), Overbought RSI, Extended % above EMA200.<br>
<b>What to do:</b> <span style="color:#ef4444;">No new entries.</span> If you hold, consider partial exits or tighten trailing stop. These are not shorts yet — that is the Short Sell layer.<br>
<b>Exception:</b> ExtN = 1 with strong StructureScore ≥ 70 = acceptable with half size.
</div>

<div style="color:#cc2244;font-weight:700;font-size:11px;margin-bottom:2px;">🔻 SHORT SELL</div>
<div style="color:#64748b;margin-bottom:10px;padding-left:12px;">
<b>What it means:</b> Bearish setup with multiple confirming signals. Distribution, trend breakdown, and weak RS all align.<br>
<b>What the engine measures:</b> ShortScore (RSI overbought + falling, RS rank in bottom 30%, SmartMoney = DISTRIBUTING, MTF bearish sync, breakdown of support).<br>
<b>Key numbers:</b> Short Score ≥ 68 = SHORT NOW; 45–67 = signal only. RSI should be falling, not just elevated.<br>
<b>What to do:</b> <span style="color:#cc2244;">SHORT NOW verdict only.</span> Use the shown SL (above recent swing high). This is a separate trade thesis from the long side — do not mix.<br>
<b>Caution:</b> In strong bull markets, short setups fail more. Check the breadth bar — if &gt;60% stocks above EMA50, short with half size only.
</div>

</div>
""", unsafe_allow_html=True)

        st.markdown('<div style="margin-bottom:6px;"></div>', unsafe_allow_html=True)

        # ── Summary bar: one metric per layer ─────────────────────────────────
        if _all_res:
            short_candidates_summary = derive_short_candidates(_all_res, scan_mode_now, vix_val)
            _sh_cnt = len([s for s in short_candidates_summary
                           if s.verdict in (SHORT_CONFIRMED, SHORT_SIGNAL)])
            _lyr_cols = st.columns(6)
            _lyr_defs = [
                ("EMERGING_MOMENTUM","🌱 Emerging",         "#a78bfa"),
                ("BREAKOUT_READY",  "⚡ Breakout Ready",   "#f59e0b"),
                ("STRONG_BUY",      "🚀 Breakout",         "#22c55e"),
                ("EXTENDED_RISKY",  "⚠ Risky",             "#ef4444"),
                ("ACCUMULATING",    "🛡 Accumulating",     "#38bdf8"),
                ("SHORT_SELL",      "🔻 Short Sell",        "#cc2244"),
            ]
            for col, (grp, lbl, clr) in zip(_lyr_cols, _lyr_defs):
                cnt = len(_buckets.get(grp, [])) if grp != "SHORT_SELL" else _sh_cnt
                col.markdown(
                    f'<div style="background:{clr}11;border:1px solid {clr}33;border-radius:8px;'
                    f'padding:8px 10px;text-align:center;">'
                    f'<div style="color:{clr};font-size:10px;font-weight:700;'
                    f'letter-spacing:.04em;">{lbl}</div>'
                    f'<div style="color:{clr};font-family:JetBrains Mono,monospace;'
                    f'font-size:22px;font-weight:700;line-height:1.2;">{cnt}</div>'
                    f'</div>', unsafe_allow_html=True
                )
            st.markdown('<div style="margin-bottom:14px;"></div>', unsafe_allow_html=True)

        # ── LAYER 1: EMERGING MOMENTUM ────────────────────────────────────────
        _layer1 = _buckets["EMERGING_MOMENTUM"][:20]
        _pre_cnt = sum(1 for r in _layer1 if r.get("Action") == "PRE-CONFIRM")
        with st.expander(
            f"🌱 EMERGING — {len(_layer1)} stocks coiling · "
            f"{_pre_cnt} PRE-CONFIRM · EM Score · RS Accel · Squeeze",
            expanded=bool(_layer1)
        ):
            if _layer1:
                _html = '<div style="display:flex;flex-wrap:wrap;gap:12px;align-items:stretch;">'
                for i, r in enumerate(_layer1):
                    _html += make_emerging_card(i, r, layer_color=_LAYER_COLORS["EMERGING_MOMENTUM"])
                _html += "</div>"
                st.markdown(_html, unsafe_allow_html=True)
                st.markdown(
                    '<div style="color:#3a3a60;font-size:10px;font-family:JetBrains Mono,monospace;'
                    'padding:5px 0 2px;text-align:center;">ⓘ Momentum building — NOT an entry signal. '
                    'Wait for phase upgrade to ENTRY / BREAKOUT before acting.</div>',
                    unsafe_allow_html=True)
            else:
                st.caption("No stocks with emerging momentum.")

        # ── LAYER 2: BREAKOUT READY ───────────────────────────────────────────
        _layer2 = _buckets["BREAKOUT_READY"][:15]
        with st.expander(
            f"⚡ BREAKOUT READY — {len(_layer2)} stocks near trigger · "
            f"Entry Proximity · Vol Confirm · HTF Align · R:R",
            expanded=bool(_layer2)
        ):
            if _layer2:
                _html = '<div style="display:flex;flex-wrap:wrap;gap:12px;align-items:stretch;">'
                for i, r in enumerate(_layer2):
                    _html += make_card(i, r, border_color=_LAYER_COLORS["BREAKOUT_READY"], show_entry=True)
                _html += "</div>"
                st.markdown(_html, unsafe_allow_html=True)
                st.markdown(
                    '<div style="color:#3a3a60;font-size:10px;font-family:JetBrains Mono,monospace;'
                    'padding:5px 0 2px;text-align:center;">ⓘ Near entry — confirm with volume and price action.</div>',
                    unsafe_allow_html=True)
            else:
                st.caption("No stocks at breakout threshold.")

        # ── LAYER 3: BREAKOUT (STRONG BUY) ───────────────────────────────────
        _layer3 = _buckets["STRONG_BUY"][:15]
        with st.expander(
            f"🚀 BREAKOUT — {len(_layer3)} confirmed leaders · "
            f"Score · ADX Trend · RS Rank · Breadth",
            expanded=bool(_layer3)
        ):
            if _layer3:
                _html = '<div style="display:flex;flex-wrap:wrap;gap:12px;align-items:stretch;">'
                for i, r in enumerate(_layer3):
                    _html += make_card(i, r, border_color=_LAYER_COLORS["STRONG_BUY"], show_entry=True)
                _html += "</div>"
                st.markdown(_html, unsafe_allow_html=True)
                st.markdown(
                    '<div style="color:#3a3a60;font-size:10px;font-family:JetBrains Mono,monospace;'
                    'padding:5px 0 2px;text-align:center;">ⓘ Price confirmed — use discipline on position sizing.</div>',
                    unsafe_allow_html=True)
            else:
                st.caption("No breakout candidates.")

        # ── LAYER 4: RISKY (EXTENDED) ─────────────────────────────────────────
        _layer4 = _buckets["EXTENDED_RISKY"][:12]
        with st.expander(
            f"⚠ RISKY — {len(_layer4)} overextended · "
            f"Ext Flags · RSI Stretch · Dist from EMA · Climax Volume",
            expanded=False
        ):
            if _layer4:
                _html = '<div style="display:flex;flex-wrap:wrap;gap:12px;align-items:stretch;">'
                for i, r in enumerate(_layer4):
                    _html += make_card(i, r, border_color=_LAYER_COLORS["EXTENDED_RISKY"], show_entry=False)
                _html += "</div>"
                st.markdown(_html, unsafe_allow_html=True)
                st.markdown(
                    '<div style="color:#3a3a60;font-size:10px;font-family:JetBrains Mono,monospace;'
                    'padding:5px 0 2px;text-align:center;">⚠ Avoid new entries — wait for base to form.</div>',
                    unsafe_allow_html=True)
            else:
                st.caption("No extended stocks found.")

        # ── LAYER 5: ACCUMULATING ─────────────────────────────────────────────
        # Engine: PCA + SmartMoney + AccumStage
        # Question answered: "Is institutional money quietly building a position?"
        # DO NOT ACT — this is a watch layer. Upgrade happens when TimingScore rises.
        _layer5 = _buckets["ACCUMULATING"][:20]
        with st.expander(
            f"🛡 ACCUMULATING — {len(_layer5)} stocks building a base · "
            f"PCA owns this layer · Vol Compression · CMF/OBV · Base Duration",
            expanded=bool(_layer5)
        ):
            if _layer5:
                _html = '<div style="display:flex;flex-wrap:wrap;gap:12px;align-items:stretch;">'
                for i, r in enumerate(_layer5):
                    _html += make_emerging_card(i, r, layer_color=_LAYER_COLORS["ACCUMULATING"])
                _html += "</div>"
                st.markdown(_html, unsafe_allow_html=True)
                st.markdown(
                    '<div style="color:#3a3a60;font-size:10px;font-family:JetBrains Mono,monospace;'
                    'padding:5px 0 2px;text-align:center;">'
                    'ⓘ <b style="color:#38bdf8;">PCA engine owns this layer</b> — institutional buying fingerprint. '
                    'Watch for StructureScore ≥ 60 + AccumStage 1C/2A to flag upgrade readiness. '
                    'Do not enter until it moves to Emerging or Breakout Ready.</div>',
                    unsafe_allow_html=True)
            else:
                st.caption("No stocks in accumulation phase.")

        # ── LAYER 6: SHORT SELL ───────────────────────────────────────────────
        short_candidates=derive_short_candidates(_all_res, scan_mode_now, vix_val)
        if short_candidates:
            sh_now=sum(1 for s in short_candidates if s.verdict==SHORT_CONFIRMED)
            sh_sig=sum(1 for s in short_candidates if s.verdict==SHORT_SIGNAL)
            sh_watch=sum(1 for s in short_candidates if s.verdict==SHORT_WATCH)
            top_shorts=[s for s in short_candidates if s.verdict in (SHORT_CONFIRMED,SHORT_SIGNAL)][:12]
            with st.expander(f"🔻 SHORT SELL — {sh_now} SHORT NOW · {sh_sig} SIGNAL · {sh_watch} WATCH · "
                             f"Short Score · RSI Weak · RS Rank · Distribution",
                             expanded=(sh_now>0)):
                if top_shorts:
                    sh_cards='<div style="display:flex;flex-wrap:wrap;gap:12px;align-items:stretch;">'
                    for i,sr in enumerate(top_shorts):
                        vc=SHORT_COLORS.get(sr.verdict,"#555577")
                        rr_c="#22c55e" if sr.risk_reward>=2 else ("#f59e0b" if sr.risk_reward>=1.5 else "#ef4444")
                        rsi_c="#ef4444" if sr.rsi_val>70 else ("#f59e0b" if sr.rsi_val>60 else "#cbd5e1")
                        bar=min(sr.short_score,100)
                        dchg=sr.day_change; dchg_s=f"+{dchg:.2f}%" if dchg>=0 else f"{dchg:.2f}%"
                        dchg_c="#22c55e" if dchg>=0 else "#ef4444"; dchg_arr="▲" if dchg>=0 else "▼"
                        hard_pills="".join(
                            f'<span style="background:#0b0b0f;border:1px solid rgba(239,68,68,0.20);'
                            f'color:#e2e8f0;padding:3px 8px;border-radius:6px;font-size:10px;'
                            f'font-weight:500;font-family:Inter,sans-serif;margin:2px;">{t}</span>'
                            for t in sr.hard_triggers)
                        soft_pills="".join(
                            f'<span style="background:#0b0b0f;border:1px solid rgba(239,68,68,0.14);'
                            f'color:#e2e8f0;padding:3px 8px;border-radius:6px;font-size:10px;'
                            f'font-weight:500;font-family:Inter,sans-serif;margin:2px;">{t}</span>'
                            for t in sr.soft_triggers)
                        ext_badge=(f'<span style="background:#0b0b0f;border:1px solid rgba(239,68,68,0.18);'
                                   f'color:#e2e8f0;padding:3px 8px;border-radius:6px;font-size:10px;'
                                   f'font-weight:500;font-family:Inter,sans-serif;margin:2px;">'
                                   f'EXT {sr.ext_n} — short fuel</span>') if sr.ext_n>=2 else ""
                        sh_cards+=(
                            f'<div style="background:#1b1113;border:1px solid {vc}55;border-radius:12px;'
                            f'overflow:hidden;min-width:240px;max-width:340px;flex:1 1 240px;">'
                            f'<div style="display:flex;align-items:center;padding:12px 16px 10px;'
                            f'border-bottom:1px solid rgba(255,255,255,0.08);gap:10px;">'
                            f'<div style="background:{vc}22;color:{vc};font-family:JetBrains Mono,monospace;'
                            f'font-size:12px;font-weight:700;padding:4px 8px;border-radius:6px;min-width:32px;text-align:center;">{i+1:02d}</div>'
                            f'<div style="font-family:Syne,sans-serif;color:#f0e8e8;font-size:16px;font-weight:700;flex:1;">{sr.symbol}</div>'
                            f'<span style="background:{vc}22;border:1px solid {vc};color:{vc};'
                            f'padding:4px 10px;border-radius:5px;font-size:11px;font-weight:700;">▼ {sr.verdict}</span>'
                            f'<span style="background:#1e1e40;color:#cbd5e1;font-family:JetBrains Mono,monospace;'
                            f'font-size:11px;padding:4px 8px;border-radius:5px;">{sr.short_score}</span>'
                            f'</div>'
                            f'<div style="display:flex;padding:12px 16px;gap:0;">'
                            f'<div style="flex:0 0 45%;padding-right:16px;border-right:1px solid #1e1e40;">'
                            f'<div style="font-family:JetBrains Mono,monospace;color:#f0e8e8;font-size:22px;font-weight:600;line-height:1;">₹{sr.current_price:,.1f}</div>'
                            f'<div style="font-family:JetBrains Mono,monospace;color:{dchg_c};font-size:13px;margin-top:4px;font-weight:500;">{dchg_s} {dchg_arr}</div>'
                            f'<div style="color:#f8fafc;font-size:11px;margin-top:3px;font-family:JetBrains Mono,monospace;">Short zone</div>'
                            f'<div style="color:#f8fafc;font-size:12px;font-weight:600;font-family:JetBrains Mono,monospace;">₹{sr.entry_zone_lo:,.1f}–₹{sr.entry_zone_hi:,.1f}</div>'
                            f'</div>'
                            f'<div style="flex:1;padding-left:16px;">'
                            f'<div style="display:flex;gap:6px;flex-wrap:wrap;">'
                            f'<span style="background:#1e1e40;padding:5px 8px;border-radius:5px;"><span style="color:#cbd5e1;font-size:9px;display:block;">SL ▲</span>'
                            f'<span style="color:#ef4444;font-family:JetBrains Mono,monospace;font-size:11px;font-weight:600;">₹{sr.stop_loss:,.1f}</span></span>'
                            f'<span style="background:#1e1e40;padding:5px 8px;border-radius:5px;"><span style="color:#cbd5e1;font-size:9px;display:block;">R:R</span>'
                            f'<span style="color:{rr_c};font-weight:700;font-size:12px;">1:{sr.risk_reward:.1f}</span></span>'
                            f'<span style="background:#1e1e40;padding:5px 8px;border-radius:5px;"><span style="color:#cbd5e1;font-size:9px;display:block;">RSI</span>'
                            f'<span style="color:{rsi_c};font-family:JetBrains Mono,monospace;font-size:11px;">{sr.rsi_val:.0f}</span></span>'
                            f'</div></div></div>'
                            f'<div style="padding:6px 16px 8px;background:#0a0808;display:flex;gap:16px;">'
                            f'<div><span style="color:#cbd5e1;font-size:9px;">T1 ▼</span>'
                            f'<div style="font-family:JetBrains Mono,monospace;color:#22aa88;font-size:11px;">₹{sr.target1:,.1f}</div></div>'
                            f'<div><span style="color:#cbd5e1;font-size:9px;">T2 ▼</span>'
                            f'<div style="font-family:JetBrains Mono,monospace;color:#22aa88;font-size:12px;font-weight:600;">₹{sr.target2:,.1f}</div></div>'
                            f'<div><span style="color:#cbd5e1;font-size:9px;">T3 ▼</span>'
                            f'<div style="font-family:JetBrains Mono,monospace;color:#22aa88;font-size:11px;">₹{sr.target3:,.1f}</div></div>'
                            f'<div style="margin-left:auto;text-align:right;">'
                            f'<span style="color:#cbd5e1;font-size:9px;">RS · {sr.sector}</span>'
                            f'<div style="color:#aaa;font-family:JetBrains Mono,monospace;font-size:11px;">RS{sr.rs_rank}</div></div>'
                            f'</div>'
                            f'<div style="padding:0 16px 6px;"><div style="background:#1e1e40;border-radius:2px;height:3px;">'
                            f'<div style="background:{vc};width:{bar}%;height:3px;border-radius:2px;"></div></div></div>'
                            f'<div style="padding:4px 16px 10px;">{hard_pills}{soft_pills}{ext_badge}</div>'
                            f'</div>'
                        )
                    sh_cards+="</div>"
                    st.markdown(sh_cards,unsafe_allow_html=True)
                    st.markdown(
                        '<div style="text-align:center;color:#94a3b8;font-size:11px;'
                        'font-family:JetBrains Mono,monospace;padding:10px 0 4px;line-height:1.6;">'
                        'ⓘ Short candidates derived from scan engine — confirm with price action.<br>'
                        '<span style="color:#fca5a5;font-weight:600;">Short selling carries elevated risk — always use disciplined SL management.</span>'
                        '</div>',unsafe_allow_html=True)

        # ── Export BUY results ──────────────────────────────────────────────
        if results:
            buy_rows=[r for r in results if r["Action"] in ("BUY","STRONG BUY")]
            if buy_rows:
                _df_exp   = pd.DataFrame(buy_rows)
                _bad_cols = [c for c in _df_exp.columns
                             if _df_exp[c].dropna().apply(lambda v: isinstance(v, (dict, list))).any()]
                csv=_df_exp.drop(columns=_bad_cols,errors="ignore").to_csv(index=False)
                ts=datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button("Export BUY results",csv,
                                   f"NSE_Scan_{st.session_state.scan_mode}_{ts}.csv","text/csv")
        elif st.session_state.get("results", []):
            st.warning("No stocks match current filters.")
        else:
            st.info("Select Universe + Mode, then press SCAN.")

# ══════════════════════════════════════════════════════════════════════════════
# BREADTH ENGINE TAB (unchanged)
# ══════════════════════════════════════════════════════════════════════════════


