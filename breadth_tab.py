"""
breadth_tab.py — Breadth Engine tab renderer.
"""
import pandas as pd
import streamlit as st
from market import compute_breadth
from scanner import derive_short_candidates
from config import SHORT_CONFIRMED, SHORT_SIGNAL

def render(all_results, vix_val, scan_mode):
    """Render this tab. Call inside `with tab_X:`."""
    all_results = st.session_state.results
    if not all_results:
        st.info("Run a scan first to see breadth data.")
    else:
        # v16.1: read pre-computed breadth from session_state (set at scan time).
        # compute_breadth on 500 stocks runs every render otherwise — removed.
        breadth = st.session_state.get("breadth") or compute_breadth(all_results)
        b_sig,b_col=breadth["breadth_signal"]
        st.markdown(
            f'<div style="background:{b_col}11;border:1px solid {b_col}33;border-radius:8px;'
            f'padding:10px 16px;margin-bottom:14px;">'
            f'<span style="font-family:Syne,sans-serif;font-size:15px;color:{b_col};">'
            f'Market Breadth: <strong>{b_sig}</strong></span></div>',
            unsafe_allow_html=True,
        )
        bm1,bm2,bm3,bm4,bm5,bm6=st.columns(6)
        bm1.metric("% Above EMA50",f'{breadth["pct_above_ema50"]}%')
        bm2.metric("% in BREAKOUT",f'{breadth["pct_breakout"]}%')
        bm3.metric("Advancing",breadth["advancing"])
        bm4.metric("Declining",breadth["declining"])
        bm5.metric("A/D Ratio",breadth["ad_ratio"])
        bm6.metric("Liquid Stocks",breadth["liquid_count"])
        gated_n=sum(1 for r in all_results if r.get("BreadthGated"))
        if gated_n:
            st.warning(f"🔵 **Breadth Gate** — {gated_n} BREAKOUT/CONT signals capped to WATCH")
        sector_data=breadth["sector_avg"]
        if sector_data:
            sec_df=pd.DataFrame([
                {"Sector":k,"Avg Score":v,"Count":sum(1 for r in all_results if r.get("Sector")==k)}
                for k,v in sorted(sector_data.items(),key=lambda x:-x[1])
            ])
            hm_html='<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:8px;">'
            for _,row in sec_df.iterrows():
                score=row["Avg Score"]
                bar_col="#22c55e" if score>=70 else ("#d97706" if score>=55 else "#ef4444")
                pct=min(100,score)
                hm_html+=(
                    f'<div style="background:#111120;border:1px solid #1e1e40;border-radius:7px;padding:10px 12px;">'
                    f'<div style="color:#e8e8f4;font-size:11px;font-weight:600;font-family:DM Sans,sans-serif;">{row["Sector"]}</div>'
                    f'<div style="color:#cbd5e1;font-size:10px;font-family:JetBrains Mono,monospace;">{int(row["Count"])} stocks</div>'
                    f'<div style="background:#1e1e40;border-radius:2px;height:4px;margin:6px 0;">'
                    f'<div style="background:{bar_col};width:{pct}%;height:4px;border-radius:2px;"></div></div>'
                    f'<div style="color:{bar_col};font-size:15px;font-weight:600;font-family:JetBrains Mono,monospace;">{score}</div></div>'
                )
            hm_html+="</div>"
            st.markdown(hm_html,unsafe_allow_html=True)
        st.markdown("---")
        dist_data={"Advancing":breadth["advancing"],"Unchanged":breadth["unchanged"],"Declining":breadth["declining"]}
        dist_colors={"Advancing":"#22c55e","Unchanged":"#d97706","Declining":"#ef4444"}
        total_shown=sum(dist_data.values())
        dist_html='<div style="display:flex;gap:8px;">'
        for label,count in dist_data.items():
            pct2=round(count/total_shown*100,1) if total_shown else 0
            col=dist_colors[label]
            dist_html+=(
                f'<div style="flex:1;background:#111120;border:1px solid {col}33;border-radius:7px;padding:12px;text-align:center;">'
                f'<div style="color:{col};font-size:22px;font-weight:600;font-family:JetBrains Mono,monospace;">{count}</div>'
                f'<div style="color:#cbd5e1;font-size:11px;font-family:DM Sans,sans-serif;">{label}</div>'
                f'<div style="color:{col};font-size:11px;font-family:JetBrains Mono,monospace;">{pct2}%</div></div>'
            )
        dist_html+="</div>"
        st.markdown(dist_html,unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DETAIL TAB (unchanged structure, adds ADX/Squeeze metrics)
# ══════════════════════════════════════════════════════════════════════════════


