"""
universe.py — Symbol universe helpers and sector map.
Re-exports SECTOR_MAP and universe helpers from config.
"""
from config import SECTOR_MAP, NIFTY50, NSE500, _SECTORS_LOOKUP

import os
import pandas as pd


def get_universe_options():
    return ["Nifty 50", "NSE 500", "Custom"]


def symbols_for_universe(universe_opt: str) -> list:
    if universe_opt == "Nifty 50":
        return list(NIFTY50)
    if universe_opt == "NSE 500":
        return list(NSE500)
    return list(NIFTY50)  # default fallback
