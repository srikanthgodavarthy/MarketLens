# Bull Sutra Pro — v18.4

Modular NSE stock scanner built on Streamlit.

## Package layout

```
bull_sutra/
├── app.py               # Streamlit entry point — run this
├── config.py            # All constants, weights, labels, colour palettes
├── universe.py          # NSE 500 / Nifty 50 symbol lists + sector map
├── data_fetch.py        # Yahoo Finance async HTTP, Parquet cache, HTF, VIX, OI
├── indicators.py        # Vectorised NumPy indicator primitives + Stage-A pre-filter
├── scoring.py           # Full scoring pipeline (phase, RS, emerging, PCA, SmartMoney…)
├── patterns.py          # VCP, AVWAP, Fib, Darvas, MTF, institutional volume, harmonics
├── market.py            # Market-regime detection, breadth engine
├── risk.py              # Stop-loss, targets, position sizing, exhaustion, staleness
├── scanner.py           # run_scan(), short-side engine
├── portfolio.py         # Open-position monitoring and exit scoring
├── persistence.py       # PostgreSQL/Supabase DB helpers, earnings cache, disk cache
├── requirements.txt
└── ui/
    ├── components.py    # make_card, make_emerging_card, colour helpers
    └── tabs/
        ├── dashboard.py
        ├── scanner_tab.py
        ├── sectors.py
        ├── breadth_tab.py
        ├── detail_tab.py
        ├── analytics_tab.py
        ├── portfolio_tab.py
        └── settings_tab.py
```

## Running

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Bug fixes applied in this refactor

| Severity | Item | Fix |
|----------|------|-----|
| 🔴 | `_rband` / `_rband_col` / `_rcause` / `_rtiming` / `_rctx` / `_rop` / `_ortho_html` computed but never rendered | **Removed** (lines 7551-7560) |
| 🔴 | `market_bullish` referenced in Supabase persist block but out of scope | **Fixed** — replaced with `"BULL" in _regime_label.upper()` |
| 🟡 | Stale comment `"Price + ReadinessBand"` | **Updated** to `"Price + stage strip"` |
| 🟡 | `_today5_reason` dead function (~67 lines) | **Removed** |
| 🟡 | `top_act` / `actionable` computed but never rendered | **Removed** |
| 🟡 | `border_color[:7]` unsafe slice on potentially-None value | **Fixed** — `(border_color or "#475569")[:7]` |
