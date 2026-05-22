"""
portfolio_tab.py — Portfolio & exit management tab.
"""
import streamlit as st
import pandas as pd
from portfolio import run_exit_scan, add_position, score_exit
from persistence import _db_save, _db_load
from components import _action_colors, _phase_color

def render(positions, vix_val, scan_mode):
    """Render this tab. Call inside `with tab_X:`."""
    st.markdown(
        '<div style="font-family:Syne,sans-serif;font-size:18px;font-weight:700;'
        'color:#e8e8f4;margin-bottom:12px;">💼 Open Positions & Exit Signals</div>',
        unsafe_allow_html=True,
    )
    with st.expander("➕ Add Position",expanded=False):
        pf1,pf2,pf3,pf4=st.columns([2,2,1,1])
        ap_sym=pf1.text_input("Symbol",key="pf_sym").upper()
        ap_entry=pf2.number_input("Entry Price ₹",min_value=0.01,value=100.0,step=0.5,key="pf_ep")
        ap_qty=pf3.number_input("Qty",min_value=1,value=100,step=1,key="pf_qty")
        ap_mode=pf4.selectbox("Mode",list(MODE_CFG.keys()),index=1,key="pf_mode")
        if st.button("Add",type="primary",key="pf_add_btn"):
            if ap_sym:
                add_position(ap_sym,ap_entry,int(ap_qty),ap_mode)
                st.success(f"Added {ap_sym}")
    positions=st.session_state.get("open_positions") or []
    if not positions:
        st.info("No open positions.")
    else:
        col_refresh,_=st.columns([1,5])
        with col_refresh:
            if st.button("🔄 Refresh Exit Signals",use_container_width=True,key="pf_refresh"):
                vix_pf,_=fetch_vix()
                with st.spinner("Scanning exits…"):
                    st.session_state["exit_results"]=run_exit_scan(positions,vix_pf)
        exit_res=st.session_state.get("exit_results",{})
        counts={EXIT_HOLD:0,EXIT_WATCH_LBL:0,EXIT_SIGNAL_LBL:0,EXIT_CONFIRM_LBL:0}
        for p in positions:
            if not isinstance(p,dict): continue
            sym=p.get("symbol")
            if not sym: continue
            er=exit_res.get(sym); lbl=er.verdict if er else EXIT_HOLD
            counts[lbl]=counts.get(lbl,0)+1
        p1,p2,p3,p4=st.columns(4)
        p1.metric("🟢 Hold",counts[EXIT_HOLD]); p2.metric("🟡 Watch",counts[EXIT_WATCH_LBL])
        p3.metric("🟠 Exit Signal",counts[EXIT_SIGNAL_LBL]); p4.metric("🔴 Exit Now",counts[EXIT_CONFIRM_LBL])
        valid_pos=[p for p in positions if isinstance(p,dict) and p.get("symbol")]

        # ── Build card HTML for a single position ─────────────────────────────
        def _pf_card_html(pos):
            sym=pos["symbol"]; er=exit_res.get(sym)
            verdict=er.verdict if er else EXIT_HOLD
            triggers=er.triggers if er else []; trail_sl=er.trailing_stop if er else None
            entry_px=pos.get("entry_price",0); curr_px=(er.current_price if (er and er.current_price) else entry_px)
            qty=pos.get("qty",0); mode_p=pos.get("mode","Swing")
            day_pct=er.day_pct if er else 0.0
            pnl_pct=(curr_px-entry_px)/entry_px*100 if entry_px else 0
            pnl_abs=round((curr_px-entry_px)*qty,0)
            pnl_col="#22c55e" if pnl_pct>=0 else "#ef4444"
            day_col="#22c55e" if day_pct>=0 else "#ef4444"
            day_str=f"+{day_pct:.2f}%" if day_pct>=0 else f"{day_pct:.2f}%"
            pnl_str=f"+{pnl_abs:,.0f}" if pnl_abs>=0 else f"{pnl_abs:,.0f}"
            vc=EXIT_COLORS.get(verdict,"#22aa55")
            sector=SECTOR_MAP.get(sym,"—")

            # Quality / Risk / Hold
            q_score=er.quality_score if er else 0
            q_label=er.quality_label if er else "—"
            r_score=er.risk_score if er else 0
            r_label=er.risk_label if er else "—"
            h_score=er.hold_score if er else 0
            h_label=er.hold_label if er else "—"
            r_mult=er.r_multiple if er else 0.0
            dd=er.drawdown_from_peak if er else 0.0
            phase=er.phase if er else ""; sm=er.smart_money_verdict if er else ""
            rs_l=er.rs_label if er else ""; days=er.days_held if er else 0

            q_col=("#22c55e" if q_score>=75 else "#84cc16" if q_score>=50
                   else "#f59e0b" if q_score>=30 else "#ef4444")
            r_col=("#ef4444" if r_score>=70 else "#ff8800" if r_score>=45
                   else "#f59e0b" if r_score>=20 else "#22c55e")
            h_col=("#22c55e" if h_score>=70 else "#84cc16" if h_score>=50
                   else "#f59e0b" if h_score>=30 else "#ef4444")

            # Phase icon
            ph_icon={"BREAKOUT":"🚀","CONT":"↗","ENTRY":"⚡","SETUP":"◎","IDLE":"–","EXIT":"↘"}.get(phase,"")
            sm_col={"MARKUP_READY":"#22c55e","ACCUMULATING":"#84cc16","ABSORBING":"#6ee7b7",
                    "NEUTRAL":"#94a3b8","DISTRIBUTING":"#ef4444"}.get(sm,"#94a3b8")
            rs_col={"LEADER":"#22c55e","IMPROVING":"#84cc16","NEUTRAL":"#94a3b8","LAGGARD":"#ef4444"}.get(rs_l,"#94a3b8")

            # Triggers rows (show top 5)
            trig_rows="".join(
                f'<div style="padding:2px 0;border-bottom:1px solid #15152a;color:#c8d0e0;font-size:9px;">{t}</div>'
                for t in triggers[:5]) or '<div style="color:#3a3a60;font-size:9px;padding:3px 0;">No signals</div>'

            trail_row=(
                f'<div style="display:flex;justify-content:space-between;padding:2px 12px;">'
                f'<span style="color:#cbd5e1;font-size:9px;">🎯 TRAIL SL</span>'
                f'<span style="font-family:JetBrains Mono,monospace;color:#f59e0b;font-size:11px;font-weight:600;">₹{trail_sl:,.2f}</span></div>'
            ) if trail_sl else ""

            def _bar(val, col, label, sub):
                return (
                    f'<div style="margin-bottom:6px;">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:2px;">'
                    f'<span style="color:#94a3b8;font-size:8px;letter-spacing:.04em;">{label}</span>'
                    f'<span style="color:{col};font-size:8px;font-weight:700;">{val} · {sub}</span></div>'
                    f'<div style="background:#1e1e40;border-radius:2px;height:4px;">'
                    f'<div style="background:{col};width:{val}%;height:4px;border-radius:2px;transition:width .3s;"></div></div></div>'
                )

            r_mult_col="#22c55e" if r_mult>=2 else "#84cc16" if r_mult>=1 else "#f59e0b" if r_mult>=0 else "#ef4444"
            r_mult_str=f"+{r_mult:.1f}R" if r_mult>=0 else f"{r_mult:.1f}R"
            dd_col="#ef4444" if dd<-10 else "#f59e0b" if dd<-5 else "#94a3b8"
            dd_str=f"{dd:.1f}%"

            return (
                f'<div style="background:#111120;border:1.5px solid {vc};border-radius:12px;'
                f'overflow:hidden;min-width:230px;max-width:320px;flex:1 1 230px;">'

                # ── Header ──────────────────────────────────────────────────────
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:10px 12px 8px;border-bottom:1px solid #1e1e40;background:#0e0e1c;">'
                f'<div><span style="font-family:Syne,sans-serif;color:#e8e8f4;font-size:15px;font-weight:700;">{sym}</span>'
                f'<div style="color:#6b7280;font-size:8.5px;">{sector} · {mode_p}'
                f'{(" · " + str(days) + "d") if days else ""}</div></div>'
                f'<span style="background:{vc}22;border:1px solid {vc};color:{vc};padding:2px 8px;'
                f'border-radius:5px;font-size:10px;font-weight:700;">{verdict}</span>'
                f'</div>'

                # ── Price row ────────────────────────────────────────────────────
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:0;padding:8px 12px;">'
                f'<div><div style="color:#64748b;font-size:7.5px;">ENTRY</div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:#94a3b8;font-size:10px;">₹{entry_px:,.2f}</div></div>'
                f'<div><div style="color:#64748b;font-size:7.5px;">CMP</div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:#e2e8f0;font-size:10px;">₹{curr_px:,.2f}</div></div>'
                f'<div><div style="color:#64748b;font-size:7.5px;">DAY</div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:{day_col};font-size:10px;font-weight:600;">{day_str}</div></div>'
                f'<div><div style="color:#64748b;font-size:7.5px;">P&L%</div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:{pnl_col};font-size:10px;font-weight:700;">{pnl_pct:+.1f}%</div></div>'
                f'</div>'

                # ── R-multiple + drawdown row ────────────────────────────────────
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0;padding:0px 12px 8px;">'
                f'<div><div style="color:#64748b;font-size:7.5px;">R-MULT</div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:{r_mult_col};font-size:11px;font-weight:700;">{r_mult_str}</div></div>'
                f'<div><div style="color:#64748b;font-size:7.5px;">FROM PEAK</div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:{dd_col};font-size:11px;font-weight:600;">{dd_str}</div></div>'
                f'<div><div style="color:#64748b;font-size:7.5px;">P&L ₹</div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:{pnl_col};font-size:11px;font-weight:700;">{pnl_str}</div></div>'
                f'</div>'

                # ── Context tags row ─────────────────────────────────────────────
                f'<div style="display:flex;gap:5px;padding:0 12px 8px;flex-wrap:wrap;">'
                + (f'<span style="background:#1e2035;border:1px solid #2a2a50;color:#94a3b8;'
                   f'font-size:8px;padding:1px 6px;border-radius:4px;">{ph_icon} {phase}</span>' if phase else "")
                + (f'<span style="background:#1e2035;border:1px solid {sm_col}44;color:{sm_col};'
                   f'font-size:8px;padding:1px 6px;border-radius:4px;">{sm}</span>' if sm else "")
                + (f'<span style="background:#1e2035;border:1px solid {rs_col}44;color:{rs_col};'
                   f'font-size:8px;padding:1px 6px;border-radius:4px;">RS {rs_l}</span>' if rs_l else "") +
                f'</div>'

                # ── Three-axis bars ──────────────────────────────────────────────
                f'<div style="padding:6px 12px 8px;background:#0d0d1a;border-top:1px solid #1a1a30;">'
                + _bar(q_score, q_col, "QUALITY", q_label)
                + _bar(100-r_score, r_col, "RISK LEVEL", r_label)
                + _bar(h_score, h_col, "HOLD CONFIDENCE", h_label) +
                f'</div>'

                + trail_row +

                # ── Signal triggers ──────────────────────────────────────────────
                f'<div style="padding:5px 12px 6px;background:#0e0e1c;border-top:1px solid #1a1a30;">'
                f'<div style="color:#475569;font-size:7.5px;margin-bottom:3px;">SIGNALS</div>'
                + trig_rows +
                f'</div>'
                f'</div>'
            )

        # ── Group positions by verdict priority then render each group ─────────
        _group_order=[EXIT_CONFIRM_LBL, EXIT_SIGNAL_LBL, EXIT_WATCH_LBL, EXIT_HOLD]
        _group_labels={
            EXIT_CONFIRM_LBL:"🔴 Exit Now",
            EXIT_SIGNAL_LBL: "🟠 Exit Signal",
            EXIT_WATCH_LBL:  "🟡 Watch",
            EXIT_HOLD:       "🟢 Hold",
        }
        for _grp in _group_order:
            _grp_pos=[p for p in valid_pos
                      if (exit_res.get(p["symbol"]).verdict if exit_res.get(p["symbol"]) else EXIT_HOLD)==_grp]
            if not _grp_pos: continue
            vc_grp=EXIT_COLORS.get(_grp,"#22aa55")
            st.markdown(
                f'<div style="color:{vc_grp};font-family:Syne,sans-serif;font-size:12px;'
                f'font-weight:700;letter-spacing:.06em;margin:14px 0 6px;">'
                f'{_group_labels[_grp]} · {len(_grp_pos)}</div>',
                unsafe_allow_html=True,
            )
            grp_html='<div style="display:flex;flex-wrap:wrap;gap:10px;align-items:stretch;">'
            for p in _grp_pos:
                grp_html+=_pf_card_html(p)
            grp_html+="</div>"
            st.markdown(grp_html, unsafe_allow_html=True)

        # ── Remove positions ───────────────────────────────────────────────────
        st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
        sym_options=[f"{p['symbol']} ({p.get('mode','—')})" for p in valid_pos]
        to_remove=st.multiselect("🗑 Remove positions", sym_options, key="pf_remove_sel")
        if to_remove and st.button("Remove selected", key="pf_remove_btn", type="primary"):
            remove_syms={s.split(" (")[0] for s in to_remove}
            st.session_state["open_positions"]=[
                p for p in st.session_state["open_positions"]
                if p.get("symbol") not in remove_syms
            ]
            _db_save("bs_positions", st.session_state["open_positions"])
            st.rerun()
