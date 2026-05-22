"""
analytics_tab.py — Signal analytics & outcome tracking tab.
"""
import pandas as pd
import streamlit as st
from risk import signal_is_stale, signal_age_label

def render(signal_log, scan_mode):
    """Render this tab. Call inside `with tab_X:`."""
    st.subheader("Signal Log & Outcome Tracking")
    log=st.session_state.signal_log
    if not log:
        st.info("No signals logged yet. Run a scan to populate.")
    else:
        log_df=pd.DataFrame(log); scan_mode_now=st.session_state.scan_mode
        log_df["stale"]=log_df.apply(
            lambda row:signal_is_stale(row["timestamp"],row.get("mode",scan_mode_now)),axis=1)
        log_df["age"]=log_df.apply(
            lambda row:signal_age_label(row["timestamp"],row.get("mode",scan_mode_now))[0],axis=1)
        total_sig=len(log_df); pending=len(log_df[log_df["outcome"]=="Pending"])
        stale_cnt=int(log_df["stale"].sum())
        wins=len(log_df[log_df["outcome"]=="Win"]); losses=len(log_df[log_df["outcome"]=="Loss"])
        win_rate=round(wins/(wins+losses)*100,1) if (wins+losses)>0 else None
        am1,am2,am3,am4=st.columns(4)
        am1.metric("Total Signals",total_sig); am2.metric("Pending",pending)
        am3.metric("Expired",stale_cnt); am4.metric("Win%",f"{win_rate}%" if win_rate else "—")
        display_cols=["timestamp","symbol","action","phase","score","confidence",
                      "rs_rank","entry","sl","t1","age","outcome","breadth_gated"]
        display_cols=[c for c in display_cols if c in log_df.columns]
        edited=st.data_editor(
            log_df[display_cols].tail(100),
            column_config={"outcome":st.column_config.SelectboxColumn("Outcome",
                            options=["Pending","Win","Loss","BE"],required=True),
                           "age":st.column_config.TextColumn("Age",disabled=True)},
            hide_index=True,use_container_width=True,
        )
        if edited is not None and len(edited)==len(log_df.tail(100)):
            for i,row in edited.iterrows():
                idx=len(log_df)-100+i
                if 0<=idx<len(log): log[idx]["outcome"]=row["outcome"]

# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO TAB (unchanged)
# ══════════════════════════════════════════════════════════════════════════════


