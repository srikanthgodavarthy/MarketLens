"""
components.py — Shared UI colour helpers and card rendering.
"""
import streamlit as st

def _action_colors(act):
    if act=="STRONG BUY": return "#f59e0b22","#f59e0b88","#f59e0b"
    if act=="BUY":        return "#22c55e1a","#22c55e66","#22c55e"
    if act=="PRE-CONFIRM":return "#8b5cf622","#8b5cf688","#a78bfa"   # v15.7
    if act=="WATCH":      return "#3b82f611","#3b82f644","#60a5fa"
    return "#cbd5e111","#cbd5e133","#cbd5e1"

def _phase_color(ph):
    return {"BREAKOUT":"#00dd88","CONT":"#22aa55","ENTRY":"#2255cc",
            "SETUP":"#b87333","IDLE":"#555577","EXIT":"#cc4444"}.get(ph,"#555577")

def _trend_color(up:bool): return "#22c55e" if up else "#ef4444"

def _rs_color(rank:int):
    if rank>=80: return "#22c55e"
    if rank>=60: return "#d97706"
    return "#94a3b8"

def _conf_color(conf:int):
    if conf>=80: return "#2ecc71"
    if conf>=60: return "#f39c12"
    if conf>=40: return "#e67e22"
    return "#e74c3c"

# ══════════════════════════════════════════════════════════════════════════════
# SPEED-8: CARD HASH for render guard
# ══════════════════════════════════════════════════════════════════════════════

