"""
detail_tab.py — Detail tab renderer.
"""
import streamlit as st
import pandas as pd
from data_fetch import fetch_vix
from components import _action_colors, _phase_color, _conf_color

def render(all_results, vix_val):
    """Render this tab. Call inside `with tab_X:`."""
    all_results=st.session_state.results
    if not all_results:
        st.info("Run a scan first.")
    else:
        sel=st.selectbox("Select stock",[r["Symbol"] for r in all_results])
        r=next((x for x in all_results if x["Symbol"]==sel),None)
        if r:
            phase=r.get("Phase",PHASE_IDLE); chg=r["%Change"]
            conf=r.get("Confidence",0); conf_lbl,conf_col=confidence_label(conf)
            phases_order=[PHASE_IDLE,PHASE_SETUP,PHASE_ENTRY,PHASE_CONT,PHASE_BRK,PHASE_EXIT]
            ph_html='<div style="display:flex;gap:5px;margin-bottom:12px;flex-wrap:wrap;">'
            for ph in phases_order:
                active=ph==phase
                bg=PHASE_COLORS[ph] if active else "#1e1e40"
                brd=f"1px solid {PHASE_COLORS[ph]}" if active else "1px solid #1e1e40"
                ph_html+=(
                    f'<div style="background:{bg};border:{brd};color:{"#e8e8f4" if active else "#cbd5e1"};'
                    f'padding:4px 12px;border-radius:5px;font-size:11px;'
                    f'font-weight:{"600" if active else "400"};font-family:DM Sans,sans-serif;">'
                    f'{ph}{"  ◀" if active else ""}</div>'
                )
            ph_html+="</div>"
            st.markdown(ph_html,unsafe_allow_html=True)
            if r.get("BreadthGated"):
                st.warning("🔵 **Breadth Gated** — action capped to WATCH due to weak market breadth.")
            d1,d2,d3,d4,d5=st.columns(5)
            d1.metric("LTP",fmt(r["LTP"]),f"{'+' if chg>=0 else ''}{chg}%")
            d2.metric("Entry ⚡",fmt(r["Entry"]))
            d3.metric("Stop Loss",fmt(r["SL"]))
            d4.metric("Score",r["Score"])
            d5.metric("Confidence",f"{conf}% ({conf_lbl})")
            t1c,t2c,t3c,r1c=st.columns(4)
            t1c.metric("T1",fmt(r["T1"])); t2c.metric("T2",fmt(r["T2"]))
            t3c.metric("T3",fmt(r["T3"]))
            risk=round(r["Entry"]-r["SL"],2) if r.get("Entry") and r.get("SL") else 0
            r1c.metric("Risk/Share",fmt(risk))
            # v15 new indicator row
            adx_c,sq_c,vc_c=st.columns(3)
            adx_c.metric("ADX",f'{r.get("ADX","—")}',
                         delta="Strong" if (r.get("ADX") or 0)>=30 else "Weak",
                         delta_color="normal" if (r.get("ADX") or 0)>=30 else "inverse")
            sq_c.metric("BB/KC Squeeze","ON 🔄" if r.get("Squeeze") else "OFF")
            vc_c.metric("Vol Contraction",f'{r.get("VolRatio","—")}',
                        delta="Compressed" if (r.get("VolRatio") or 1)<0.75 else "Normal",
                        delta_color="normal" if (r.get("VolRatio") or 1)<0.75 else "off")
            # v15.3 category score breakdown
            st.markdown("**Score Breakdown (Category Weights)**")
            _cT=r.get("CatT",0); _cM=r.get("CatM",0); _cS=r.get("CatS",0)
            _cV=r.get("CatV",0); _cQ=r.get("CatQ",0)
            _cat_cols=st.columns(5)
            for _col,_lbl,_val,_mx,_tip in [
                (_cat_cols[0],"TREND",    _cT,30,"EMA stack · HTF · regime · cross"),
                (_cat_cols[1],"MOMENTUM", _cM,20,"RSI · 1M/3M/6M mom"),
                (_cat_cols[2],"STRUCTURE",_cS,20,"Phase · Fib zone · HH · RS rank"),
                (_cat_cols[3],"VOLUME",   _cV,15,"Vol ratio · ADX strength"),
                (_cat_cols[4],"QUALITY",  _cQ,15,"Squeeze · Vol contraction · Clean"),
            ]:
                _pct=int(_val/_mx*100) if _mx>0 else 0
                _col.metric(_lbl,f"{_val:.1f}/{_mx}",f"{_pct}%",
                            delta_color="normal" if _pct>=60 else ("off" if _pct>=30 else "inverse"))
            # v15.6 Pre-Confirmation Accumulation breakdown
            st.markdown("**Pre-Confirmation Accumulation (PCA)** — buying-pressure layer")
            _pca_score = r.get("PCAScore", 0); _pca_lbl = r.get("PCALabel", "NONE")
            _pca_col = {"ACCUMULATING":"#22c55e","BUILDING":"#38bdf8",
                        "FORMING":"#a78bfa","WEAK":"#64748b","NONE":"#374151"}.get(_pca_lbl,"#374151")
            st.markdown(
                f'<div style="background:{_pca_col}11;border:1px solid {_pca_col}33;border-radius:7px;'
                f'padding:7px 14px;margin-bottom:8px;display:flex;align-items:center;gap:12px;">'
                f'<span style="background:{_pca_col};color:#0a0a0f;padding:2px 10px;border-radius:4px;'
                f'font-family:JetBrains Mono,monospace;font-size:12px;font-weight:700;">'
                f'{_pca_lbl} · {_pca_score}</span>'
                f'<span style="color:#94a3b8;font-size:10px;">Detects institutional buying BEFORE price confirms</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            pca_c1,pca_c2,pca_c3,pca_c4=st.columns(4)
            pca_c1.metric("Relative CMF",    f'{r.get("PCACMFRel",0):.0f}/15',
                          delta="Active" if r.get("PCACMFRel",0)>=8 else None,
                          delta_color="normal" if r.get("PCACMFRel",0)>=8 else "off")
            pca_c2.metric("Vol Cmp Seq",     f'{r.get("PCAVolCmpSeq",0):.0f}/15',
                          delta="Active" if r.get("PCAVolCmpSeq",0)>=8 else None,
                          delta_color="normal" if r.get("PCAVolCmpSeq",0)>=8 else "off")
            pca_c3.metric("Hidden Accum",    f'{r.get("PCAHiddenAccum",0):.0f}/15',
                          delta="Active" if r.get("PCAHiddenAccum",0)>=8 else None,
                          delta_color="normal" if r.get("PCAHiddenAccum",0)>=8 else "off")
            pca_c4.metric("Effort/Result",   f'{r.get("PCAEffortResult",0):.0f}/15',
                          delta="Active" if r.get("PCAEffortResult",0)>=8 else None,
                          delta_color="normal" if r.get("PCAEffortResult",0)>=8 else "off")
            pca_c5,pca_c6,pca_c7,_ = st.columns(4)
            pca_c5.metric("NR Persistence",  f'{r.get("PCARangeCont",0):.0f}/10',
                          delta="Active" if r.get("PCARangeCont",0)>=6 else None,
                          delta_color="normal" if r.get("PCARangeCont",0)>=6 else "off")
            pca_c6.metric("Failed Breakdown", f'{r.get("PCAFailedBrkdn",0):.0f}/15',
                          delta="Active" if r.get("PCAFailedBrkdn",0)>=8 else None,
                          delta_color="normal" if r.get("PCAFailedBrkdn",0)>=8 else "off")
            pca_c7.metric("Vol Asymmetry",   f'{r.get("PCAVolAsym",0):.0f}/15',
                          delta="Active" if r.get("PCAVolAsym",0)>=8 else None,
                          delta_color="normal" if r.get("PCAVolAsym",0)>=8 else "off")
            # v15.1/15.2 pattern signals
            st.markdown("**Pattern Signals** *(enriched post-scan)*")
            _pat=r.get("Patterns",{})
            _vcp_d=_pat.get("vcp",{}); _avwap_d=_pat.get("avwap",{})
            _fibq_d=_pat.get("fib_quality",{}); _vdu_d=_pat.get("vol_dryup",{})
            _rvol_d=_pat.get("rel_vol",{}); _darv_d=_pat.get("darvas",{})
            pc1,pc2,pc3,pc4,pc5,pc6=st.columns(6)
            pc1.metric("VCP",f'{_vcp_d.get("vcp_grade","—")} ({_vcp_d.get("n_contractions",0)}×)',
                       delta="Confirmed" if _vcp_d.get("detected") else None,
                       delta_color="normal" if _vcp_d.get("detected") else "off")
            _av=_avwap_d.get("avwap"); _avp=_avwap_d.get("pct_above",0)
            pc2.metric("Anch.VWAP",f'₹{_av:,.1f}' if _av else "—",
                       delta=f'{_avp:+.1f}%',
                       delta_color="normal" if _avwap_d.get("price_above") else "inverse")
            pc3.metric("Fib Pullback",_fibq_d.get("grade","—"),
                       delta=f'Q:{_fibq_d.get("quality",0)}',
                       delta_color="normal" if _fibq_d.get("quality",0)>=60 else "off")
            pc4.metric("Vol Dry-up",f'{"×"*int(_vdu_d.get("intensity",0)) or "—"} ({_vdu_d.get("bars",0)}b)',
                       delta="Active" if _vdu_d.get("dry_up") else None,
                       delta_color="normal" if _vdu_d.get("dry_up") else "off")
            pc5.metric("Rel.Volume",_rvol_d.get("label","—"),
                       delta=f'{_rvol_d.get("rel_vol_pct",50):.0f}th · {_rvol_d.get("ratio",1):.1f}×',
                       delta_color="normal" if _rvol_d.get("rel_vol_pct",50)>=65 else "off")
            _dbrk=_darv_d.get("breakout"); _din=_darv_d.get("in_box")
            _dtop=_darv_d.get("box_top",0); _dbot=_darv_d.get("box_bottom",0)
            pc6.metric("Darvas","BREAKOUT" if _dbrk else ("IN BOX" if _din else "—"),
                       delta=f'₹{_dbot:,.0f}–₹{_dtop:,.0f}' if _dtop else None,
                       delta_color="normal" if _dbrk else "off")
            st.markdown("---")
            with st.expander("Position Sizing",expanded=True):
                _acct_size=st.session_state.get("account_size",500000)
                _risk_pct=st.session_state.get("risk_pct",0.02)
                _max_cap_pct=st.session_state.get("max_capital_pct",0.20)
                ps=position_size(account_size=_acct_size,entry=r["Entry"],sl=r["SL"],
                                 atr_val=r.get("ATR",risk),atr_mean=r.get("ATR_Mean",risk),
                                 vix_val=vix_val,risk_pct=_risk_pct,max_capital_pct=_max_cap_pct)
                ps1,ps2,ps3,ps4=st.columns(4)
                ps1.metric("Suggested Qty",ps["final_qty"])
                ps2.metric("Capital Used",fmt(ps["capital_used"]))
                ps3.metric("Max Loss",fmt(ps["max_loss"]))
                ps4.metric("Risk per Share",fmt(risk))
            ext_n=r.get("ExtN",0); ext_labels=r.get("ExtLabels",[]); ext_flags=r.get("ExtFlags",{})
            if ext_n==0:
                st.success("✅ No extension/exhaustion signals — structure is clean.")
            else:
                flag_desc={
                    "rsi_overheat":"Stock ran up too fast — buyers exhausted. Wait for cooldown.",
                    "atr_extension":"Today's range unusually large — possible blow-off.",
                    "parabolic":"Price jumped far more than normal in 3 bars. Hard to sustain.",
                    "ema_distance":"Price stretched way above its average. Pullback likely.",
                    "climactic_volume":"Huge volume spike with long upper wick — potential distribution.",
                    "mom_exhaustion":"Price rising but buying pressure quietly weakening.",
                    "bearish_div":"New high, but momentum didn't confirm it.",
                }
                with st.expander(f"⚠ {ext_n} Caution Signal{'s' if ext_n>1 else ''} — "
                                  f"{'DO NOT enter' if ext_n>=3 else 'Reduce size'}",expanded=True):
                    for fk,fa in ext_flags.items():
                        if fa:
                            ec="#ef4444" if ext_n>=3 else "#f59e0b"
                            st.markdown(f'<div style="color:{ec};font-size:12px;padding:3px 0;">'
                                        f'▸ <strong>{fk.replace("_"," ").title()}</strong> — '
                                        f'{flag_desc.get(fk,"")}</div>',unsafe_allow_html=True)
            info_cols=st.columns(4)
            info_cols[0].metric("RSI",r.get("RSI","—"))
            info_cols[1].metric("RS Rank",f'{r.get("RS_Rank",50)}/100')
            info_cols[2].metric("Liq (₹Cr/d)",r.get("AvgTradedCr","—"))
            info_cols[3].metric("Raw RS Diff",f"{r.get('RS',0):+.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS TAB (unchanged)
# ══════════════════════════════════════════════════════════════════════════════


