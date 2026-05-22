"""
sectors.py — Sectors tab renderer.
"""
import streamlit as st
import pandas as pd
from market import compute_breadth

def render(all_results, scan_mode):
    """Render this tab. Call inside `with tab_X:`."""
    # v16.1: mark tab as visited on first render
    st.session_state["tab_sectors_loaded"] = True
    _sec_det_full = st.session_state.get("breadth", {}).get("sector_detail", {})
    _res_sec      = st.session_state.get("results", [])

    if not _sec_det_full:
        st.markdown(
            '<div style="color:#475569;font-size:12px;font-family:JetBrains Mono,monospace;'
            'padding:20px 0;text-align:center;">Run a scan to populate sector scores.</div>',
            unsafe_allow_html=True
        )
    else:
        # ── Sort control ────────────────────────────────────────────────────
        _sort_by = st.selectbox(
            "Sort sectors by",
            ["Ready Score (desc)", "PCA Score (desc)", "EM Score (desc)",
             "% Advancing (desc)", "Actionable (desc)", "Stock Count (desc)"],
            key="sec_sort_by", label_visibility="visible"
        )
        _sort_key = {
            "Ready Score (desc)":  lambda x: x[1]["avg_ready"],
            "PCA Score (desc)":    lambda x: x[1]["avg_structure"],
            "EM Score (desc)":     lambda x: x[1]["avg_timing"],
            "% Advancing (desc)":  lambda x: x[1]["pct_adv"],
            "Actionable (desc)":   lambda x: x[1]["actionable"],
            "Stock Count (desc)":  lambda x: x[1]["count"],
        }.get(_sort_by, lambda x: x[1]["avg_ready"])

        _secs_full = sorted(_sec_det_full.items(), key=_sort_key, reverse=True)

        # ── Table header ────────────────────────────────────────────────────
        st.markdown(
            '<div style="display:grid;'
            'grid-template-columns:220px 90px 90px 90px 90px 1fr;'
            'gap:0;background:#0d1320;border:1px solid #1e2a3a;border-radius:8px 8px 0 0;'
            'padding:8px 12px;font-family:JetBrains Mono,monospace;font-size:9px;'
            'color:#475569;letter-spacing:.06em;font-weight:700;text-transform:uppercase;">'
            '<div>Sector</div>'
            '<div style="text-align:center;">Ready</div>'
            '<div style="text-align:center;">PCA</div>'
            '<div style="text-align:center;">EM</div>'
            '<div style="text-align:center;">Advancing</div>'
            '<div style="padding-left:8px;">Actionable Stocks</div>'
            '</div>',
            unsafe_allow_html=True
        )

        for _idx, (_sname, _sd) in enumerate(_secs_full):
            _sr   = _sd["avg_ready"]
            _ss   = _sd["avg_structure"]    # PCA / structure score
            _st2  = _sd["avg_timing"]       # EM / timing score
            _pa   = _sd["pct_adv"]
            _cnt  = _sd["count"]
            _ac   = _sd["actionable"]
            _ldr  = _sd["leaders"]          # top 3 actionable stocks
            _bk   = _sd["breakouts"]

            # Row colour theme
            _sc2  = "#22c55e" if _sr >= 65 else "#f59e0b" if _sr >= 45 else "#ef4444" if _sr < 30 else "#38bdf8"
            _pca_c= "#22c55e" if _ss >= 60 else "#f59e0b" if _ss >= 40 else "#ef4444"
            _em_c = "#22c55e" if _st2 >= 60 else "#f59e0b" if _st2 >= 40 else "#ef4444"
            _ad_c = "#22c55e" if _pa >= 60 else "#f59e0b" if _pa >= 40 else "#ef4444"

            # Verdict badge
            if _ss >= 60 and _st2 >= 60:   _verdict, _vc = "READY",    "#22c55e"
            elif _ss >= 50 and _st2 >= 50: _verdict, _vc = "WATCH",    "#f59e0b"
            elif _ss >= 50:                _verdict, _vc = "BUILDING", "#38bdf8"
            elif _st2 >= 50:               _verdict, _vc = "COILING",  "#a78bfa"
            else:                          _verdict, _vc = "DORMANT",  "#475569"

            # ── Build actionable stock cards (mini chips) ─────────────────
            _stock_chips = ""
            for _ld in _ldr:
                _lc   = "#22c55e" if _ld["action"] == "STRONG BUY" else ("#f59e0b" if _ld["action"] == "BUY" else "#a78bfa")
                _lchg = _ld.get("chg", 0)
                _lcc  = "#22c55e" if _lchg >= 0 else "#ef4444"
                _ltp_val = _ld.get("ltp", 0)
                _stock_chips += (
                    f'<div style="background:#0b1422;border:1px solid #1e2a3a;border-left:2px solid {_lc};'
                    f'border-radius:5px;padding:4px 8px;display:flex;flex-direction:column;min-width:90px;flex-shrink:0;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;gap:4px;">'
                    f'<span style="font-family:Syne,sans-serif;color:#f1f5f9;font-size:10px;font-weight:700;">{_ld["sym"]}</span>'
                    f'<span style="background:{_lc}22;color:{_lc};font-size:7px;font-weight:700;'
                    f'padding:1px 4px;border-radius:2px;">{_ld["action"][:3]}</span>'
                    f'</div>'
                    f'<div style="display:flex;justify-content:space-between;margin-top:2px;">'
                    f'<span style="font-family:JetBrains Mono,monospace;color:#94a3b8;font-size:9px;">₹{_ltp_val:,.0f}</span>'
                    f'<span style="font-family:JetBrains Mono,monospace;color:{_lcc};font-size:9px;">'
                    f'{"+" if _lchg>=0 else ""}{_lchg:.1f}%</span>'
                    f'</div>'
                    f'<div style="margin-top:2px;">'
                    f'<div style="background:#1e2a3a;border-radius:1px;height:2px;">'
                    f'<div style="background:{_lc};width:{min(int(_ld["ready"]),100)}%;height:2px;border-radius:1px;"></div></div>'
                    f'</div>'
                    f'</div>'
                )

            if not _stock_chips:
                _stock_chips = '<span style="color:#334155;font-size:9px;font-family:JetBrains Mono,monospace;">—</span>'

            _row_bg = "#0d1320" if _idx % 2 == 0 else "#0a0f1c"
            _border_t = f"border-top:1px solid #1e2a3a;" if _idx > 0 else ""
            _is_last = _idx == len(_secs_full) - 1
            _border_rad = "border-radius:0 0 8px 8px;" if _is_last else ""

            st.markdown(
                f'<div style="display:grid;grid-template-columns:220px 90px 90px 90px 90px 1fr;'
                f'gap:0;background:{_row_bg};{_border_t}{_border_rad}'
                f'padding:10px 12px;align-items:center;">'

                # Sector name + verdict badge
                f'<div>'
                f'<div style="font-family:Syne,sans-serif;color:#e2e8f0;font-size:11px;font-weight:600;">{_sname}</div>'
                f'<div style="margin-top:3px;display:flex;gap:4px;align-items:center;">'
                f'<span style="background:{_vc}22;border:1px solid {_vc}44;color:{_vc};'
                f'font-size:7px;font-weight:700;padding:1px 5px;border-radius:3px;">{_verdict}</span>'
                f'<span style="color:#334155;font-size:8px;">{_cnt} stocks'
                + (f' · {_bk}🔥' if _bk else '') +
                f'</span></div></div>'

                # Ready Score
                f'<div style="text-align:center;">'
                f'<div style="color:{_sc2};font-family:JetBrains Mono,monospace;font-size:16px;font-weight:700;">{_sr:.0f}</div>'
                f'<div style="background:#1e2a3a;border-radius:2px;height:3px;margin:3px 8px 0;">'
                f'<div style="background:{_sc2};width:{min(int(_sr),100)}%;height:3px;border-radius:2px;"></div></div>'
                f'</div>'

                # PCA Score
                f'<div style="text-align:center;">'
                f'<div style="color:{_pca_c};font-family:JetBrains Mono,monospace;font-size:16px;font-weight:700;">{_ss:.0f}</div>'
                f'<div style="background:#1e2a3a;border-radius:2px;height:3px;margin:3px 8px 0;">'
                f'<div style="background:{_pca_c};width:{min(int(_ss),100)}%;height:3px;border-radius:2px;"></div></div>'
                f'</div>'

                # EM Score
                f'<div style="text-align:center;">'
                f'<div style="color:{_em_c};font-family:JetBrains Mono,monospace;font-size:16px;font-weight:700;">{_st2:.0f}</div>'
                f'<div style="background:#1e2a3a;border-radius:2px;height:3px;margin:3px 8px 0;">'
                f'<div style="background:{_em_c};width:{min(int(_st2),100)}%;height:3px;border-radius:2px;"></div></div>'
                f'</div>'

                # % Advancing
                f'<div style="text-align:center;">'
                f'<div style="color:{_ad_c};font-family:JetBrains Mono,monospace;font-size:16px;font-weight:700;">{_pa:.0f}%</div>'
                f'<div style="color:#334155;font-size:8px;margin-top:2px;">{_ac} actionable</div>'
                f'</div>'

                # Actionable stock cards
                f'<div style="padding-left:8px;display:flex;gap:6px;flex-wrap:wrap;">'
                + _stock_chips +
                f'</div>'

                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS TAB
# ══════════════════════════════════════════════════════════════════════════════


