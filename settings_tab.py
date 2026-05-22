"""
settings_tab.py — Settings tab renderer.
"""
import streamlit as st
from config import MODE_CFG

def render():
    """Render this tab. Call inside `with tab_X:`."""
    st.subheader("Scanner Settings")
    sc1,sc2=st.columns(2)
    with sc1:
        st.session_state.min_liq_cr=st.slider(
            "Min Liquidity (₹ Cr daily traded value)",1.0,50.0,
            float(st.session_state.min_liq_cr),1.0)
        st.session_state.phase_filter=st.selectbox(
            "Phase Filter (Scanner)",
            ["All Phases","ENTRY","SETUP","CONT","BREAKOUT","IDLE","EXIT"],
            index=["All Phases","ENTRY","SETUP","CONT","BREAKOUT","IDLE","EXIT"].index(
                st.session_state.get("phase_filter","All Phases")))
        st.session_state.show_illiquid=st.checkbox(
            "Show illiquid stocks (below liquidity floor)",value=st.session_state.show_illiquid)
        st.markdown("---"); st.markdown("**Position Sizing**")
        st.session_state.account_size=st.number_input(
            "Account Size (₹)",min_value=10000,max_value=10_000_000,
            value=int(st.session_state.account_size),step=10000)
        st.session_state.risk_pct=st.slider(
            "Risk per trade (%)",0.5,5.0,float(st.session_state.risk_pct*100),0.5)/100.0
        st.session_state.max_capital_pct=st.slider(
            "Max capital per trade (% of account)",5,50,
            int(st.session_state.max_capital_pct*100),5)/100.0
        st.caption(
            f"Current cap: ₹{st.session_state.account_size*st.session_state.max_capital_pct:,.0f} "
            f"per position ({int(st.session_state.max_capital_pct*100)}% of account)"
        )
        st.markdown("---"); st.markdown("**v15 Cache**")
        col_c1,col_c2=st.columns(2)
        with col_c1:
            cache_size=sum(f.stat().st_size for f in _CACHE_DIR.glob("*.parquet"))
            st.metric("Cache size",f"{cache_size/1e6:.1f} MB")
            st.metric("Cached files",len(list(_CACHE_DIR.glob("*.parquet"))))
        with col_c2:
            if st.button("🗑 Clear cache",use_container_width=True):
                for f in _CACHE_DIR.glob("*.parquet"): f.unlink(missing_ok=True)
                for f in _CACHE_DIR.glob("cold_start_*.flag"): f.unlink(missing_ok=True)
                st.success("Cache cleared — next scan will do a full fetch.")
    with sc2:
        st.markdown("**v15 Speed Architecture**")
        st.markdown("""
| Component | v14 | v15 |
|-----------|-----|-----|
| HTTP | yfinance | aiohttp direct |
| Fetch | Sequential batches | 64 concurrent |
| History | Re-download every scan | Parquet cache + live tail |
| Pre-filter | None | Stage-A EMA vectorized |
| Indicators | Per-symbol Pandas | Batch numpy (N×T) |
| ADX | ✗ | ✓ |
| Squeeze | ✗ | ✓ |
| Vol contraction | ✗ | ✓ |
""")
        st.markdown("**Action Thresholds**")
        st.markdown("""
| Score | Action |
|-------|--------|
| ≥ 75 | STRONG BUY |
| ≥ 58 | BUY |
| ≥ 42 | WATCH |
| < 42 | SKIP |
""")

# ══════════════════════════════════════════════════════════════════════════════
# SCAN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

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
            vix_val=vix_val,min_liq_cr=st.session_state.min_liq_cr,
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
    st.session_state.scan_time=(
        datetime.now().strftime("%H:%M:%S")+
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
            [mode_opt, json.dumps({"bull": market_bullish, "label": _regime_label,
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


