# app.py
from __future__ import annotations

"""
Streamlit UI (yfinance + optional Groq)

What this UI does:
- Sidebar inputs (ticker, lookback, chart options, outlook options)
- KPI tiles
- Candlestick (or close-line) chart + moving averages
- Volume chart
- Future outlook chart (Monte Carlo quantile bands + sample paths)
- Agent report (Groq if configured, otherwise fallback)
- CSV download
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from agent_engine import run_agentic_research


# -----------------------------
# Page config + styling
# -----------------------------

st.set_page_config(page_title="Quant Research Agent", page_icon="Q", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.1rem; padding-bottom: 2rem; }
      div[data-testid="stMetric"] { background: rgba(255,255,255,0.03); padding: 12px; border-radius: 14px; }
      .small-note { opacity: 0.8; font-size: 0.92rem; }
      .card { background: rgba(255,255,255,0.03); padding: 14px; border-radius: 14px; }
      .section-title { margin-top: 0.2rem; margin-bottom: 0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Quant Research Agent")
st.caption("Data source: yfinance. Report: Groq (optional).")


# -----------------------------
# Sidebar controls
# -----------------------------

with st.sidebar:
    st.header("Inputs")

    ticker = st.text_input("Ticker", value="NVDA").strip().upper()
    lookback_years = st.selectbox("Lookback (years)", options=[1, 2, 3, 5, 10], index=3)

    st.markdown("---")
    st.subheader("Chart")
    chart_type = st.radio("Type", options=["Candlestick", "Close line"], index=0)
    show_mas = st.checkbox("Show moving averages", value=True)

    st.markdown("---")
    st.subheader("Future outlook")
    horizon_days = st.slider("Horizon (trading days)", min_value=20, max_value=252, value=60, step=5)
    n_sims = st.selectbox("Simulations", [500, 1000, 2000, 3000, 5000], index=3)
    show_sample_paths = st.checkbox("Show sample paths", value=True)

    st.markdown("---")
    run = st.button("Run research", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown(
        '<div class="small-note">Tip: If a ticker fails, try an exchange suffix (example: 7203.T).</div>',
        unsafe_allow_html=True,
    )


# -----------------------------
# Caching
# -----------------------------

@st.cache_data(show_spinner=False)
def _run_cached(ticker_: str, lookback_years_: int, horizon_days_: int, n_sims_: int) -> dict:
    out = run_agentic_research(
        ticker=ticker_,
        lookback_years=lookback_years_,
        horizon_days=horizon_days_,
        n_sims=n_sims_,
    )
    # Keep a copy to make Streamlit cache stable
    out2 = dict(out)
    out2["df"] = out["df"].copy()
    return out2


# -----------------------------
# Plot helpers
# -----------------------------

def _plot_price(df: pd.DataFrame, chart_type_: str, show_mas_: bool) -> go.Figure:
    fig = go.Figure()

    if chart_type_ == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC",
            )
        )
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))

    if show_mas_:
        for col in ["MA20", "MA50", "MA200"]:
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col))

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Date",
        yaxis_title="Price",
    )
    return fig


def _plot_volume(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"))
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Date",
        yaxis_title="Shares",
    )
    return fig


def _plot_outlook(outlook: dict, show_paths: bool) -> go.Figure:
    """
    Plot:
    - median forecast
    - 25-75 band
    - 10-90 band
    - optional sample simulated paths
    """
    bands = outlook["bands"]
    H = int(outlook["horizon_days"])
    x = list(range(1, H + 1))

    q10 = np.array(bands["q10"], dtype=float)
    q25 = np.array(bands["q25"], dtype=float)
    q50 = np.array(bands["q50"], dtype=float)
    q75 = np.array(bands["q75"], dtype=float)
    q90 = np.array(bands["q90"], dtype=float)

    fig = go.Figure()

    # Optional sample paths (lightweight visual)
    if show_paths:
        for p in outlook["sample_paths"]:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=p,
                    mode="lines",
                    line=dict(width=1),
                    opacity=0.12,
                    name="Path",
                    showlegend=False,
                )
            )

    # 10-90 band (shaded)
    fig.add_trace(go.Scatter(x=x, y=q10, mode="lines", line=dict(width=1), name="10th"))
    fig.add_trace(go.Scatter(x=x, y=q90, mode="lines", line=dict(width=1), name="90th", fill="tonexty"))

    # 25-75 band (shaded)
    fig.add_trace(go.Scatter(x=x, y=q25, mode="lines", line=dict(width=1), name="25th"))
    fig.add_trace(go.Scatter(x=x, y=q75, mode="lines", line=dict(width=1), name="75th", fill="tonexty"))

    # Median
    fig.add_trace(go.Scatter(x=x, y=q50, mode="lines", line=dict(width=2), name="Median"))

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Trading days ahead",
        yaxis_title="Simulated price",
    )
    return fig


# -----------------------------
# Main
# -----------------------------

if not run:
    st.info("Use the left sidebar to run the research.")
    st.stop()

if not ticker:
    st.error("Enter a ticker symbol.")
    st.stop()

with st.spinner("Fetching data and building report..."):
    try:
        out = _run_cached(ticker, int(lookback_years), int(horizon_days), int(n_sims))
    except Exception as e:
        st.error(str(e))
        st.stop()

df: pd.DataFrame = out["df"]
kpis = out["kpis"]
tech = out["tech"]
outlook = out["outlook"]
report = out["report"]
audit = out["audit"]

# KPI tiles
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Last Close", f"{kpis['last_close']:.2f}")
c2.metric("CAGR", f"{kpis['cagr']*100:.2f}%")
c3.metric("Ann. Return", f"{kpis['ann_return']*100:.2f}%")
c4.metric("Ann. Vol", f"{kpis['ann_vol']*100:.2f}%")
c5.metric("Sharpe (rf=0)", f"{kpis['sharpe_0rf']:.2f}")
c6.metric("Max Drawdown", f"{kpis['max_drawdown']*100:.2f}%")

st.markdown("")

tab_dash, tab_report, tab_export, tab_debug = st.tabs(["Dashboard", "Report", "Export", "Debug"])

with tab_dash:
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader(f"{ticker} price")
        st.plotly_chart(_plot_price(df, chart_type, show_mas), use_container_width=True)
        st.plotly_chart(_plot_volume(df), use_container_width=True)

        st.subheader("Future outlook (stochastic simulation)")
        st.plotly_chart(_plot_outlook(outlook, show_paths=show_sample_paths), use_container_width=True)
        st.caption("Outlook is a simulation based on historical return distribution. It is not a prediction.")

    with right:
        st.subheader("Snapshot")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(f"Date range: {kpis['start_date']} to {kpis['end_date']}")
        st.write(f"Rows: {kpis['days']}")
        st.write(f"Above 200D MA: {'Yes' if tech['above_ma200'] else 'No'}")
        st.write(f"50D > 200D: {'Yes' if tech['ma50_gt_ma200'] else 'No'}")

        if tech.get("ma20") is not None:
            st.write(f"MA20: {tech['ma20']:.2f}")
        if tech.get("ma50") is not None:
            st.write(f"MA50: {tech['ma50']:.2f}")
        if tech.get("ma200") is not None:
            st.write(f"MA200: {tech['ma200']:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("")
        st.subheader("Outlook stats")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write(f"Horizon: {outlook['horizon_days']} trading days")
        st.write(f"Simulations: {outlook['n_sims']}")
        st.write(f"Implied drift (ann): {outlook['mu_ann']*100:.2f}%")
        st.write(f"Implied vol (ann): {outlook['vol_ann']*100:.2f}%")
        st.write(f"P(final > spot): {outlook['prob_finish_up']*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

with tab_report:
    st.subheader("Agent report")
    st.markdown(report)

with tab_export:
    st.subheader("Export")
    export_df = df.copy()
    export_df.index.name = "Date"

    st.download_button(
        "Download CSV",
        export_df.to_csv().encode("utf-8"),
        file_name=f"{ticker}_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("Preview")
    st.dataframe(export_df.tail(80), use_container_width=True)

with tab_debug:
    st.subheader("Debug")
    st.json(audit)
