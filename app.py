"""
app.py — Streamlit entry point for Bull Sutra Pro.
Run with: streamlit run app.py
"""
import streamlit as st
from pathlib import Path

import time
from datetime import datetime
from config import (
    MODE_CFG, VIX_CALM, VIX_CAUTION, VIX_STRESS,
    _CACHE_DIR, LIQUIDITY_MIN_CR, PHASE_BRK, _SECTORS, NSE500, NIFTY50,
)
from data_fetch import (
    fetch_vix, fetch_nifty, fetch_indices, fetch_async,
    batch_incremental_fetch, _is_market_open, _cold_start_needed,
    _AIOHTTP_OK, _POLARS_OK, _PARQUET_OK,
)
from indicators import stage_a_prefilter, action_label
from scoring import score_stock
from scanner import run_scan
from market import get_cached_regime, compute_breadth
from persistence import (
    _db_load, _db_save, _save_scan_cache,
    _load_scan_cache, _compute_top5, _get_earnings_cached,
)
from universe import symbols_for_universe
import dashboard as tab_dash
import scanner_tab as tab_scan
import sectors as tab_sec
import breadth_tab as tab_brd
import detail_tab as tab_det
import analytics_tab as tab_ana
import portfolio_tab as tab_pf
import settings_tab as tab_stg


# ══════════════════════════════════════════════════════════════════════════════
# v15.8-FIX: PRE-ENTRY CHECKLIST SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="font-family:Syne,sans-serif;font-size:13px;font-weight:700;'
        'color:#f59e0b;margin-bottom:10px;letter-spacing:.05em;'
        'text-transform:uppercase;">Pre-Entry Checklist</div>',
        unsafe_allow_html=True,
    )
    _checks = [
        ("ltp_near_entry",    "LTP within 0.5% of Entry price"),
        ("nifty_flat_rising", "Nifty flat or rising right now"),
        ("htf_bullish",       "HTFUp = True (check Detail tab)"),
        ("ext_n_ok",          "ExtN is 0 or 1 (no exhaustion)"),
        ("no_earnings",       "No earnings in next 7 days"),
        ("no_resistance",     "No major resistance within 2%"),
        ("size_checked",      "Position size reviewed"),
    ]
    _all_ok = True
    for _ck, _cl in _checks:
        _v = st.checkbox(_cl, key=f"chk_{_ck}")
        if not _v: _all_ok = False
    if _all_ok:
        st.success("✅ All clear — proceed")
    else:
        _rem = sum(1 for _ck,_ in _checks if not st.session_state.get(f"chk_{_ck}",False))
        st.warning(f"⚠ {_rem} item{'s' if _rem>1 else ''} unchecked")
    if st.button("Reset", key="chk_reset", use_container_width=True):
        for _ck,_ in _checks:
            st.session_state[f"chk_{_ck}"] = False
        st.rerun()
    st.markdown("---")
    # Show any active earnings alerts from last scan
    _em = st.session_state.get("earnings_map", {})
    if _em:
        st.markdown(
            '<div style="font-size:11px;font-weight:600;color:#fca5a5;margin-bottom:4px;">'
            '⚠ Results upcoming (14d)</div>', unsafe_allow_html=True
        )
        for _s, _d in list(_em.items())[:8]:
            st.markdown(
                f'<div style="font-size:10px;font-family:JetBrains Mono,monospace;'
                f'color:#f87171;">{_s} · {_d}</div>', unsafe_allow_html=True
            )
st.markdown(
    '''<div style="font-family:Syne,sans-serif;font-size:28px;font-weight:700;
    letter-spacing:-1px;color:#e8e8f4;padding:8px 0 4px;">
    <span style="color:#f59e0b;">&#x1F402;</span> BULL SUTRA
    <span style="font-size:13px;color:#cbd5e1;font-family:JetBrains Mono,monospace;
    font-weight:400;">PRO · v16.0 ⚡</span></div>''',
    unsafe_allow_html=True,
)

_UNIVERSE_OPTIONS = (
    ["NSE 500"]+[k for k in _SECTORS.keys() if k!="Nifty 500"]
    if _SECTORS else ["NSE 500","Nifty 50"]
)

gc1,gc2,gc3,gc4,gc5,gc6 = st.columns([2,2,1,1,2,2])
with gc1:
    universe_opt=st.selectbox("Universe",_UNIVERSE_OPTIONS,index=0,label_visibility="visible")
with gc2:
    mode_opt=st.radio("Mode",["Swing","Intraday","Positional"],horizontal=True)
with gc3:
    scan_btn=st.button("SCAN",type="primary",use_container_width=True)
with gc4:
    # SPEED-8: live refresh toggle
    live_refresh=st.toggle("⚡ Live",value=st.session_state.get("live_refresh_enabled",False),
                            help="Auto-refresh live tail every 60s during market hours")
    st.session_state["live_refresh_enabled"]=live_refresh
with gc5:
    filter_opt=st.selectbox("Filter",
        ["BUY + STRONG BUY","STRONG BUY only","WATCH + BUY","PRE-CONFIRM","All Results"],
        label_visibility="collapsed", key="filter_opt_topbar")
with gc6:
    search_q=st.text_input("Search symbol",placeholder="e.g. RELIANCE",
                        label_visibility="collapsed", key="search_q_topbar")

# ── v15.8: Unified view — no mode toggle, both sections always shown ───────────
em_min_score = st.session_state.get("em_min_score", 35)

vix_val,vix_label=fetch_vix()
vix_color={"CALM":"#22c55e","CAUTION":"#f59e0b","STRESS":"#ef4444","UNKNOWN":"#cbd5e1"}.get(vix_label,"#cbd5e1")
vix_text_color={"CALM":"#14532d","CAUTION":"#78350f","STRESS":"#7f1d1d","UNKNOWN":"#374151"}.get(vix_label,"#374151")

# Speed indicators
aiohttp_badge  = ("⚡ async HTTP" if _AIOHTTP_OK   else "yfinance fallback")
polars_badge   = ("🔷 Polars"     if _POLARS_OK    else "")
parquet_badge  = ("📦 Parquet cache" if _PARQUET_OK else "")
cold_flag_info = "🌅 cold-start" if _cold_start_needed(mode_opt) else "♻️ incremental"
speed_badges   = " · ".join(filter(None,[aiohttp_badge,polars_badge,parquet_badge,cold_flag_info]))

st.markdown(
    f'<div style="background:{vix_color}18;border:1px solid {vix_color}44;'
    f'border-radius:7px;padding:7px 14px;margin:6px 0;display:flex;'
    f'align-items:center;gap:12px;font-family:JetBrains Mono,monospace;flex-wrap:wrap;">'
    f'<span style="background:{vix_color};color:{vix_text_color};padding:2px 8px;'
    f'border-radius:4px;font-size:11px;font-weight:700;">VIX '
    f'{vix_val if vix_val else "—"} · {vix_label}</span>'
    f'<span style="color:#475569;font-size:10px;">{speed_badges}</span>'
    +(f'<span style="color:#ef4444;font-size:11px;">⚠ High VIX: STRONG BUY blocked · targets compressed</span>'
      if (vix_val and vix_val>=VIX_STRESS) else "")
    +(f'<span style="color:#f59e0b;font-size:11px;">⚡ Elevated VIX: targets compressed · SL widened</span>'
      if (vix_val and VIX_CAUTION<=vix_val<VIX_STRESS) else "")
    +'</div>',
    unsafe_allow_html=True,
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_dashboard,tab_scanner,tab_sectors,tab_breadth,tab_detail,tab_portfolio,tab_analytics,tab_settings = st.tabs(
    ["🏠 Dashboard","Scanner","📊 Sectors","Breadth Engine","Detail","💼 Portfolio","Analytics","Settings"]
)

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD TAB

if scan_btn:
    if universe_opt=="NSE 500":
        symbols=NSE500
    elif _SECTORS and universe_opt in _SECTORS:
        _sec_syms=_SECTORS[universe_opt]
        symbols=NSE500 if _sec_syms is None else list(_sec_syms)
    else:
        symbols=NIFTY50

    n=len(symbols)
    prog=st.progress(0); stat=st.empty()
    t0=time.time()
    with st.spinner(f"Scanning {universe_opt} ({n} stocks) · {mode_opt}…"):
        results,rejected,liq_skipped=run_scan(
            symbols,mode_opt,prog,stat,
            vix_val=vix_val,min_liq_cr=st.session_state.get("min_liq_cr", 10),
        )
    elapsed=time.time()-t0
    st.session_state.results=results
    st.session_state.rejected=rejected
    st.session_state.liq_skipped=liq_skipped
    st.session_state.scan_mode=mode_opt
    # Save breadth for Dashboard + Sectors tabs (was never persisted before)
    from collections import defaultdict as _dd
    _breadth_now = compute_breadth(results)
    st.session_state["breadth"] = _breadth_now
    from datetime import timezone, timedelta
    _IST = timezone(timedelta(hours=5, minutes=30))
    st.session_state.scan_time=(
        datetime.now(_IST).strftime("%H:%M:%S IST")+
        f" ({universe_opt} · {mode_opt} · {elapsed:.0f}s)"
    )
    # v16.1: pre-compute Top5 once at scan time and persist it
    st.session_state["top5"] = _compute_top5(results)

    # v16.1: persist last-scan metadata so it survives page refresh
    _regime_label = results[0].get("Regime", "—") if results else "—"
    st.session_state["last_scan_meta"] = {
        "universe" : universe_opt,
        "mode"     : mode_opt,
        "elapsed"  : round(elapsed, 1),
        "ts"       : datetime.now().isoformat(),
        "n_scored" : len(results),
        "n_total"  : n,
        "rejected" : rejected,
        "regime"   : _regime_label,
    }
    # v16.1: write results + meta to disk so they survive a full page refresh
    _save_scan_cache(results, st.session_state["last_scan_meta"])

    # v16.1: persist regime to Supabase so cron worker + other sessions share it
    try:
        _conn = _db_conn(); _cur = _conn.cursor()
        _db_ensure(_cur); _db_ensure_worker_tables(_cur); _conn.commit()
        _cur.execute(
            "INSERT INTO bs_regime_cache (mode, data) VALUES (%s, %s)",
            [mode_opt, json.dumps({"bull": "BULL" in _regime_label.upper(), "label": _regime_label,
                           "ts": datetime.utcnow().isoformat()})]
        )
        _cur.execute("""DELETE FROM bs_regime_cache WHERE id NOT IN (
                            SELECT id FROM bs_regime_cache ORDER BY ts DESC LIMIT 5)""")
        _conn.commit(); _cur.close(); _conn.close()
    except Exception:
        pass

    # v16.1: reset tab-loaded flags so stale tab content re-renders
    for _k in ("tab_sectors_loaded", "tab_breadth_loaded",
                "tab_analytics_loaded", "tab_detail_loaded"):
        st.session_state[_k] = False

    st.rerun()   # force dashboard + all tiles to render with fresh session_state
    ts=datetime.now().isoformat(); validity_h=MODE_CFG[mode_opt]["validity_hours"]
    for r in results:
        if r.get("Action") in ("BUY","STRONG BUY"):
            st.session_state.signal_log.append({
                "timestamp":ts,"symbol":r["Symbol"],"action":r["Action"],
                "phase":r.get("Phase"),"score":r["Score"],
                "confidence":r.get("Confidence",0),"rs_rank":r.get("RS_Rank",50),
                "entry":r.get("Entry"),"sl":r.get("SL"),"t1":r.get("T1"),
                "ltp_at_signal":r.get("LTP"),"mode":mode_opt,
                "validity_hours":validity_h,"outcome":"Pending",
                "breadth_gated":r.get("BreadthGated",False),
            })
    prog.empty(); stat.empty()
    survivors_count=n-rejected
    st.success(
        f"✅ {len(results)} scored · {survivors_count} survived Stage-A · "
        f"{rejected} Stage-A filtered · {liq_skipped} illiquid · "
        f"⏱ {elapsed:.1f}s"
    )
    # v16.1: earnings no longer fetched here — loaded lazily from Supabase
    # cache on first render of the Scanner tab (see _get_earnings_cached below).
    st.rerun()   # ← ADD THIS — forces dashboard to re-render with fresh data   
# ── SPEED-8: Live refresh during market hours ──────────────────────────────────
if (live_refresh and _is_market_open()
        and st.session_state.results
        and st.session_state.scan_mode):
    import streamlit as _st
    # Auto-rerun every 60 seconds via st.rerun inside a time check
    if "last_live_refresh" not in st.session_state:
        st.session_state["last_live_refresh"] = time.time()
    elapsed_since = time.time() - st.session_state.get("last_live_refresh", 0)
    if elapsed_since >= 60:
        st.session_state["last_live_refresh"] = time.time()
        st.info("⚡ Live: refreshing latest candle…")
        # Refresh only LTP for existing results (lightweight)
        syms = [r["Symbol"] for r in st.session_state.results[:30]]
        cfg  = MODE_CFG[st.session_state.scan_mode]
        live = fetch_async(syms,"1d",cfg["interval"],concurrency=20)
        for r in st.session_state.results:
            sym = r["Symbol"]
            df  = live.get(sym)
            if df is not None and not df.empty:
                ltp       = float(df["Close"].iloc[-1])
                # Use the prior bar's close as base — NOT the cached LTP — so
                # %Change always reflects the day move from yesterday's close,
                # not a tick-to-tick delta between two live polls.
                prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else None
                r["LTP"]   = round(ltp, 2)
                if prev_close:
                    r["%Change"] = round((ltp - prev_close) / prev_close * 100, 2)
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# SCANNER TAB
# ══════════════════════════════════════════════════════════════════════════════



# ── Tab dispatch ───────────────────────────────────────────────────────────────
vix_val, vix_label = fetch_vix()

with tab_dashboard:
    tab_dash.render(
        st.session_state.get("results", []),
        st.session_state.get("breadth", {}),
        st.session_state.get("last_scan_meta"),
    )
with tab_scanner:
    tab_scan.render(
        st.session_state.get("results", []),
        vix_val, vix_label,
        st.session_state.get("scan_mode", "Swing"),
        st.session_state.get("signal_log", []),
    )
with tab_sectors:
    if st.session_state.get("tab_sectors_loaded") or st.session_state.get("results"):
        st.session_state["tab_sectors_loaded"] = True
        tab_sec.render(st.session_state.get("results", []), st.session_state.get("scan_mode", "Swing"))
with tab_breadth:
    if st.session_state.get("tab_breadth_loaded") or st.session_state.get("results"):
        st.session_state["tab_breadth_loaded"] = True
        tab_brd.render(st.session_state.get("results", []), vix_val, st.session_state.get("scan_mode", "Swing"))
with tab_detail:
    if st.session_state.get("tab_detail_loaded") or st.session_state.get("results"):
        st.session_state["tab_detail_loaded"] = True
        tab_det.render(st.session_state.get("results", []), vix_val)
with tab_analytics:
    if st.session_state.get("tab_analytics_loaded") or st.session_state.get("signal_log"):
        st.session_state["tab_analytics_loaded"] = True
        tab_ana.render(st.session_state.get("signal_log", []), st.session_state.get("scan_mode", "Swing"))
with tab_portfolio:
    tab_pf.render(
        st.session_state.get("open_positions", []),
        vix_val,
        st.session_state.get("scan_mode", "Swing"),
    )
with tab_settings:
    tab_stg.render()
