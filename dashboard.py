"""
dashboard.py — Dashboard tab renderer.
"""
import streamlit as st
from datetime import datetime

from config import (
    MODE_CFG, _CACHE_DIR, PHASE_BRK, PHASE_CONT, PHASE_SETUP,
    REGIME_TREND, REGIME_EXPANSION, REGIME_DISTRIBUTION, REGIME_PANIC, REGIME_ROTATION,
    _REGIME_ADJUSTMENTS,
)
from data_fetch import fetch_indices, fetch_oi_data, _is_market_open, fetch_vix, fetch_nifty
from market import get_cached_regime
from persistence import _compute_top5, _save_scan_cache, _load_scan_cache
from components import _action_colors, _phase_color, _conf_color

def render(all_results, breadth, last_scan_meta):
    """Render the Dashboard tab. Call inside `with tab_dashboard:`."""
    _res_dash    = st.session_state.get("results", [])
    _bread_dash  = st.session_state.get("breadth", {})
    _idx_n       = st.session_state.get("index_nifty")
    _idx_s       = st.session_state.get("index_sensex")
    _mr          = st.session_state.get("master_regime", {})

    # ── Master Regime Banner ────────────────────────────────────────────────
    if _mr:
        _mreg   = _mr.get("regime", REGIME_TREND)
        _mlab   = _mr.get("regime_label", "")
        _madj   = _mr.get("adjustments", {})
        _mreg_c = {
            REGIME_EXPANSION:    "#f59e0b",
            REGIME_TREND:        "#22c55e",
            REGIME_ROTATION:     "#38bdf8",
            REGIME_DISTRIBUTION: "#ef4444",
            REGIME_PANIC:        "#dc2626",
        }.get(_mreg, "#475569")
        _sz_pct  = int(_madj.get("size_pct", 1.0) * 100)
        _tgt_m   = _madj.get("target_mult", 1.0)
        _pc_mode = _madj.get("preconfirm", "normal")
        st.markdown(
            f'<div style="background:#0a0f1a;border:1px solid {_mreg_c}44;border-left:4px solid {_mreg_c};'
            f'border-radius:8px;padding:10px 16px;margin-bottom:12px;'
            f'display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">'
            f'<div>'
            f'<div style="color:{_mreg_c};font-family:JetBrains Mono,monospace;font-size:13px;font-weight:700;">'
            f'REGIME: {_mreg}</div>'
            f'<div style="color:#94a3b8;font-size:10px;margin-top:2px;">{_mlab}</div>'
            f'</div>'
            f'<div style="display:flex;gap:16px;">'
            f'<div style="text-align:center;">'
            f'<div style="color:#475569;font-size:8px;">POSITION SIZE</div>'
            f'<div style="color:{_mreg_c};font-family:JetBrains Mono,monospace;font-size:14px;font-weight:700;">{_sz_pct}%</div>'
            f'</div>'
            f'<div style="text-align:center;">'
            f'<div style="color:#475569;font-size:8px;">TARGET MULT</div>'
            f'<div style="color:{_mreg_c};font-family:JetBrains Mono,monospace;font-size:14px;font-weight:700;">{_tgt_m:.2f}×</div>'
            f'</div>'
            f'<div style="text-align:center;">'
            f'<div style="color:#475569;font-size:8px;">PRE-CONFIRM</div>'
            f'<div style="color:{_mreg_c};font-family:JetBrains Mono,monospace;font-size:14px;font-weight:700;">{_pc_mode.upper()}</div>'
            f'</div>'
            f'</div></div>',
            unsafe_allow_html=True
        )

    # ── Index tiles ─────────────────────────────────────────────────────────
    def _idx_tile(name, snap, color):
        if not snap:
            return (f'<div style="background:#0d1117;border:1px solid #1e2a3a;border-radius:10px;'
                    f'padding:14px 18px;text-align:center;">'
                    f'<div style="color:#475569;font-size:10px;">{name}</div>'
                    f'<div style="color:#334155;font-size:11px;margin-top:4px;">Run scan first</div></div>')
        chg_c = "#22c55e" if snap["chg"] >= 0 else "#ef4444"
        arr   = "▲" if snap["chg"] >= 0 else "▼"
        return (f'<div style="background:#0d1117;border:1px solid #1e2a3a;border-left:3px solid {chg_c};'
                f'border-radius:10px;padding:14px 18px;text-align:center;">'
                f'<div style="color:#64748b;font-size:10px;font-weight:600;letter-spacing:.05em;">{name}</div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:#f1f5f9;font-size:22px;'
                f'font-weight:700;line-height:1.2;margin-top:4px;">{snap["last"]:,.2f}</div>'
                f'<div style="color:{chg_c};font-family:JetBrains Mono,monospace;font-size:11px;margin-top:2px;">'
                f'{arr} {abs(snap["chg"]):.2f}%</div></div>')

    _dc1, _dc2, _dc3, _dc4 = st.columns(4)
    _bs = _bread_dash.get("breadth_signal", ("—","#475569"))
    _bs_lbl, _bs_col = _bs if isinstance(_bs, tuple) else ("—","#475569")
    _dc1.markdown(_idx_tile("NIFTY 50", _idx_n, "#38bdf8"), unsafe_allow_html=True)
    _dc2.markdown(_idx_tile("SENSEX",   _idx_s, "#a78bfa"), unsafe_allow_html=True)
    _dc3.markdown(
        f'<div style="background:#0d1117;border:1px solid #1e2a3a;border-left:3px solid {_bs_col};'
        f'border-radius:10px;padding:14px 18px;text-align:center;">'
        f'<div style="color:#64748b;font-size:10px;font-weight:600;">MARKET BREADTH</div>'
        f'<div style="font-family:JetBrains Mono,monospace;color:{_bs_col};font-size:22px;font-weight:700;'
        f'line-height:1.2;margin-top:4px;">{_bs_lbl}</div>'
        f'<div style="color:#475569;font-size:10px;margin-top:2px;">'
        f'{_bread_dash.get("pct_above_ema50","—")}% above EMA50</div></div>',
        unsafe_allow_html=True
    )
    _dc4.markdown(
        f'<div style="background:#0d1117;border:1px solid #1e2a3a;border-left:3px solid #f59e0b;'
        f'border-radius:10px;padding:14px 18px;text-align:center;">'
        f'<div style="color:#64748b;font-size:10px;font-weight:600;">SCANNED</div>'
        f'<div style="font-family:JetBrains Mono,monospace;color:#f1f5f9;font-size:22px;font-weight:700;'
        f'line-height:1.2;margin-top:4px;">{len(_res_dash)}</div>'
        f'<div style="color:#475569;font-size:10px;margin-top:2px;">'
        f'{_bread_dash.get("advancing","—")} adv · {_bread_dash.get("declining","—")} dec</div></div>',
        unsafe_allow_html=True
    )

    st.markdown('<div style="margin-bottom:12px;"></div>', unsafe_allow_html=True)

    # ── v16.1: Last scan memory banner ────────────────────────────────────────
    _meta = st.session_state.get("last_scan_meta")
    if _meta:
        from datetime import datetime as _dt
        try:
            _scan_ts   = _dt.fromisoformat(_meta["ts"]).strftime("%d %b %H:%M")
        except Exception:
            _scan_ts   = _meta.get("ts", "—")
        _regime_col = (
            "#22c55e" if "BULL" in str(_meta.get("regime","")).upper()
            else "#ef4444" if "BEAR" in str(_meta.get("regime","")).upper()
            else "#f59e0b"
        )
        st.markdown(
            f'<div style="background:#0b1422;border:1px solid #1e2a3a;border-radius:8px;'
            f'padding:8px 14px;margin-bottom:10px;display:flex;gap:20px;align-items:center;'
            f'font-family:JetBrains Mono,monospace;font-size:10px;flex-wrap:wrap;">'
            f'<span style="color:#64748b;">🕐 Last scan</span>'
            f'<span style="color:#f1f5f9;font-weight:700;">{_scan_ts}</span>'
            f'<span style="color:#64748b;">Universe</span>'
            f'<span style="color:#38bdf8;">{_meta.get("universe","—")}</span>'
            f'<span style="color:#64748b;">Mode</span>'
            f'<span style="color:#a78bfa;">{_meta.get("mode","—")}</span>'
            f'<span style="color:#64748b;">Scored</span>'
            f'<span style="color:#f1f5f9;">{_meta.get("n_scored","—")}/{_meta.get("n_total","—")}</span>'
            f'<span style="color:#64748b;">Regime</span>'
            f'<span style="color:{_regime_col};font-weight:700;">{_meta.get("regime","—")}</span>'
            f'<span style="color:#64748b;">⏱</span>'
            f'<span style="color:#f1f5f9;">{_meta.get("elapsed","—")}s</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── Today's 5 (mirrored from scanner tab, compact version) ─────────────
    st.markdown(
        '<div style="font-family:Syne,sans-serif;font-size:14px;font-weight:700;'
        'color:#f59e0b;margin-bottom:6px;">🎯 TODAY\'S 5</div>',
        unsafe_allow_html=True
    )
    # v16.1: read pre-computed Top5 persisted at scan time — never re-derived here.
    # This guarantees Dashboard and Scanner tab always show the same list, and the
    # list survives page refresh via disk cache.
    _t5_dash = st.session_state.get("top5", [])

    if not _t5_dash:
        st.markdown('<div style="color:#475569;font-size:11px;padding:8px 0;">No picks pass all filters. Run a scan first or check the Scanner tab.</div>',
                    unsafe_allow_html=True)
    else:
        _d5_html = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px;">'
        for _rk, _r in enumerate(_t5_dash, 1):
            _sym  = _r["Symbol"]
            _sc   = _r.get("StructureScore", 0)
            _tc   = _r.get("TimingScore", 0)
            _act  = _r.get("Action", "")
            _ltp  = _r.get("LTP", 0)
            _sl   = _r.get("SL", 0)
            _t1   = _r.get("T1", 0)
            _chg  = _r.get("%Change", 0)
            _em   = _r.get("EmLabel", "QUIET")
            _entry= _r.get("Entry") or _ltp
            _rr   = round((_t1-_entry)/(_entry-_sl), 1) if (_t1 and _sl and _entry and (_entry-_sl)>0) else 0
            _ac   = "#22c55e" if _act == "STRONG BUY" else "#f59e0b"
            _cc   = "#22c55e" if _chg >= 0 else "#ef4444"
            _rrc  = "#22c55e" if _rr >= 2 else "#f59e0b" if _rr >= 1.5 else "#64748b"
            _em_c = {"IGNITING":"#f59e0b","BUILDING":"#22c55e","COILING":"#a78bfa","LATENT":"#38bdf8","QUIET":"#475569"}.get(_em,"#475569")
            _d5_html += (
                f'<div style="background:#0b1422;border:1px solid #1e2a3a;border-left:3px solid {_ac};'
                f'border-radius:8px;padding:10px 12px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<span style="font-family:Syne,sans-serif;color:#f1f5f9;font-size:13px;font-weight:700;">#{_rk} {_sym}</span>'
                f'<span style="color:{_em_c};font-size:8px;font-weight:600;">{_em}</span></div>'
                f'<div style="font-family:JetBrains Mono,monospace;color:#f1f5f9;font-size:16px;'
                f'font-weight:700;margin-top:4px;">₹{_ltp:,.2f}'
                f'<span style="color:{_cc};font-size:10px;margin-left:6px;">{"+" if _chg>=0 else ""}{_chg:.2f}%</span></div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;margin-top:6px;">'
                f'<div><div style="color:#475569;font-size:7px;">SL</div>'
                f'<div style="color:#f87171;font-family:JetBrains Mono,monospace;font-size:9px;">₹{_sl:,.0f}</div></div>'
                f'<div><div style="color:#475569;font-size:7px;">T1</div>'
                f'<div style="color:#4ade80;font-family:JetBrains Mono,monospace;font-size:9px;">₹{_t1:,.0f}</div></div>'
                f'<div><div style="color:#475569;font-size:7px;">R:R</div>'
                f'<div style="color:{_rrc};font-family:JetBrains Mono,monospace;font-size:9px;">{_rr:.1f}×</div></div>'
                f'</div>'
                f'<div style="display:flex;gap:4px;margin-top:5px;align-items:center;">'
                f'<span style="color:#38bdf8;font-size:7px;">S{_sc:.0f}</span>'
                f'<div style="flex:1;background:#1e2a3a;border-radius:1px;height:3px;">'
                f'<div style="background:#38bdf8;width:{min(int(_sc),100)}%;height:3px;border-radius:1px;"></div></div>'
                f'<span style="color:#a78bfa;font-size:7px;">T{_tc:.0f}</span>'
                f'<div style="flex:1;background:#1e2a3a;border-radius:1px;height:3px;">'
                f'<div style="background:#a78bfa;width:{min(int(_tc),100)}%;height:3px;border-radius:1px;"></div></div>'
                f'</div></div>'
            )
        _d5_html += '</div>'
        st.markdown(_d5_html, unsafe_allow_html=True)

    st.markdown('<div style="margin-bottom:14px;"></div>', unsafe_allow_html=True)

    # ── Sector heatmap (mini version on Dashboard) ─────────────────────────
    _sec_det = _bread_dash.get("sector_detail", {})
    if _sec_det:
        st.markdown(
            '<div style="font-family:Syne,sans-serif;font-size:14px;font-weight:700;'
            'color:#a78bfa;margin-bottom:6px;">📊 SECTOR HEAT</div>',
            unsafe_allow_html=True
        )
        # Sort sectors by avg ReadinessScore desc
        _secs_sorted = sorted(_sec_det.items(), key=lambda x: x[1]["avg_ready"], reverse=True)
        _sh_html = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:6px;">'
        for _sname, _sd in _secs_sorted[:12]:   # top 12 sectors, best first
            _sr  = _sd["avg_ready"]
            _ss  = _sd["avg_structure"]
            _st2 = _sd["avg_timing"]
            _pa  = _sd["pct_adv"]
            _cnt = _sd["count"]
            _bk  = _sd["breakouts"]
            # Border/headline colour by overall score
            _sc2 = "#22c55e" if _sr >= 65 else "#f59e0b" if _sr >= 45 else "#ef4444" if _sr < 30 else "#38bdf8"
            # Per-axis colours
            _pca_c = "#22c55e" if _ss >= 60 else "#f59e0b" if _ss >= 40 else "#ef4444"
            _em_c  = "#22c55e" if _st2 >= 60 else "#f59e0b" if _st2 >= 40 else "#ef4444"
            _ad2   = "#22c55e" if _pa >= 60 else "#f59e0b" if _pa >= 40 else "#ef4444"
            # One-word verdict from S+T combination
            if _ss >= 60 and _st2 >= 60:   _verdict, _vc = "READY",    "#22c55e"
            elif _ss >= 50 and _st2 >= 50: _verdict, _vc = "WATCH",    "#f59e0b"
            elif _ss >= 50:                _verdict, _vc = "BUILDING", "#38bdf8"
            elif _st2 >= 50:               _verdict, _vc = "COILING",  "#a78bfa"
            else:                          _verdict, _vc = "DORMANT",  "#475569"
            _sh_html += (
                f'<div style="background:#0b1422;border:1px solid #1e2a3a;'
                f'border-top:2px solid {_sc2};border-radius:6px;padding:8px 10px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                f'<div style="font-family:JetBrains Mono,monospace;color:#f1f5f9;font-size:10px;'
                f'font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{_sname[:16]}</div>'
                f'<span style="background:{_vc}22;border:1px solid {_vc}55;color:{_vc};'
                f'font-size:7px;font-weight:700;padding:1px 5px;border-radius:3px;white-space:nowrap;">{_verdict}</span>'
                f'</div>'
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-top:4px;">'
                f'<span style="color:{_sc2};font-family:JetBrains Mono,monospace;font-size:18px;font-weight:700;'
                f'line-height:1;">{_sr:.0f}</span>'
                f'<span style="color:#475569;font-size:8px;">{_cnt} stocks</span></div>'
                f'<div style="display:flex;gap:6px;margin-top:5px;flex-wrap:wrap;">'
                f'<span style="color:{_pca_c};font-size:8px;font-weight:600;">PCA:{_ss:.0f}</span>'
                f'<span style="color:{_em_c};font-size:8px;font-weight:600;">EM:{_st2:.0f}</span>'
                f'<span style="color:{_ad2};font-size:8px;">{_pa:.0f}%↑</span>'
                + (f'<span style="color:#f59e0b;font-size:8px;">{_bk}🔥</span>' if _bk else '')
                + f'</div></div>'
            )
        _sh_html += '</div>'
        st.markdown(_sh_html, unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#475569;font-size:11px;padding:8px 0;">Run a scan to populate sector data.</div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTORS TAB — Table view: Sector | Ready | PCA | EM | Advancing | Actionable
# ══════════════════════════════════════════════════════════════════════════════

