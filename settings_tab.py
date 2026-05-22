"""
settings_tab.py — Settings tab renderer.
"""
import streamlit as st
from config import MODE_CFG, _CACHE_DIR

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
