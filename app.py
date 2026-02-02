from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from agent_engine import run_agentic_research

st.set_page_config(page_title="Quant Research Agent", layout="wide")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Space+Grotesk:wght@500;600&display=swap');

      .stApp {
        background: radial-gradient(circle at top, #0b1f3a 0%, #070b14 45%, #05070d 100%);
        color: #e6f1ff;
      }

      .block-container { max-width: 1200px; padding-top: 1.5rem; }

      h1, h2, h3, h4 {
        font-family: "Space Grotesk", "Inter", sans-serif;
        letter-spacing: 0.4px;
      }

      html, body, [class*="css"] {
        font-family: "Inter", sans-serif;
      }

      [data-testid="stSidebar"] {
        border-right: 1px solid rgba(148, 163, 184, 0.25);
        background: rgba(8, 14, 26, 0.65);
        backdrop-filter: blur(12px);
      }

      [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.35);
      }

      .stButton > button {
        background: linear-gradient(120deg, #1d4ed8 0%, #38bdf8 100%);
        color: #020617;
        border: none;
        border-radius: 999px;
        padding: 0.6rem 1.4rem;
        font-weight: 700;
        box-shadow: 0 10px 30px rgba(56, 189, 248, 0.35);
      }

      .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 34px rgba(56, 189, 248, 0.45);
      }

      [data-testid="stCaptionContainer"] {
        color: rgba(226, 232, 240, 0.72);
      }

      .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        letter-spacing: 0.3px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Quant Research Agent")
st.caption("LangGraph workflow · GPT-5 synthesis · yfinance data · optional LlamaIndex / SEC EDGAR context")

plot_theme = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0"),
)

with st.sidebar:
    st.subheader("Inputs")
    ticker = st.text_input("Ticker", "NVDA")
    lookback = st.selectbox("Lookback (years)", [1, 3, 5, 10], index=2)
    horizon = st.slider("Outlook horizon (days)", 20, 252, 60)
    sims = st.selectbox("Monte Carlo simulations", [1000, 3000, 5000], index=1)

    st.divider()
    st.subheader("Context")
    docs_dir = st.text_input("Local docs dir (optional)", "./data/docs")
    use_sec = st.toggle("If no local docs, pull SEC filing", value=True)
    forms = st.multiselect("Filing types", ["10-K", "10-Q", "S-1"], default=["10-K", "10-Q", "S-1"])
    sec_max_chars = st.slider("SEC text max length", 3000, 20000, 12000, step=1000)

    st.divider()
    run = st.button("Run research", type="primary")

if not run:
    st.info("Set inputs and click Run research.")
    st.stop()

with st.spinner("Running agent workflow..."):
    out = run_agentic_research(
        ticker=ticker,
        lookback_years=int(lookback),
        horizon_days=int(horizon),
        n_sims=int(sims),
        docs_dir=docs_dir,
        use_sec=bool(use_sec),
        sec_forms=forms if forms else ["10-K", "10-Q", "S-1"],
        sec_max_chars=int(sec_max_chars),
    )

df = out["df"]
kpis = out["kpis"]
outlook = out["outlook"]
report = out["report"]
trace = out["trace"]
retrieved = out.get("retrieved", {})

cols = st.columns(6)
cols[0].metric("Last Close", f"{kpis['last_close']:.2f}")
cols[1].metric("CAGR", f"{kpis['cagr']*100:.2f}%")
cols[2].metric("Ann Return", f"{kpis['ann_return']*100:.2f}%")
cols[3].metric("Ann Vol", f"{kpis['ann_vol']*100:.2f}%")
cols[4].metric("Sharpe", f"{kpis['sharpe_0rf']:.2f}")
cols[5].metric("Max DD", f"{kpis['max_drawdown']*100:.2f}%")

tab1, tab2, tab3 = st.tabs(["Dashboard", "Report", "Workflow"])

with tab1:
    left, right = st.columns([2, 1])

    with left:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                name="Close",
                line=dict(color="#38bdf8", width=2.6),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MA20"],
                name="MA20",
                line=dict(color="#22c55e", width=1.6, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MA50"],
                name="MA50",
                line=dict(color="#f97316", width=1.8, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MA200"],
                name="MA200",
                line=dict(color="#a855f7", width=2.2),
            )
        )
        fig.update_layout(
            height=520,
            title=f"{ticker.upper()} Price + Moving Averages",
            **plot_theme,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Context source")
        st.write(f"Enabled: {retrieved.get('enabled', False)}")
        st.write(f"Source: {retrieved.get('source', 'n/a')}")
        if retrieved.get("source") == "sec" and retrieved.get("enabled"):
            st.write(f"Form: {retrieved.get('form')}")
            st.write(f"Filing date: {retrieved.get('filing_date')}")
            url = retrieved.get("url")
            if url:
                st.link_button("Open filing", url)
        else:
            reason = retrieved.get("reason")
            if reason:
                st.caption(f"Reason: {reason}")

    st.subheader("Future outlook (Monte Carlo simulation)")
    x = list(range(1, outlook["horizon_days"] + 1))
    b = outlook["bands"]

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=x,
            y=b["q90"],
            name="90th",
            line=dict(color="rgba(56, 189, 248, 0.9)", width=1.4),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=x,
            y=b["q10"],
            name="10th",
            line=dict(color="rgba(56, 189, 248, 0.4)", width=1.4),
            fill="tonexty",
            fillcolor="rgba(56, 189, 248, 0.18)",
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=x,
            y=b["q50"],
            name="Median",
            line=dict(color="#f8fafc", width=2.4),
        )
    )
    fig2.update_layout(
        height=420,
        title="Simulated price distribution bands",
        **plot_theme,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.markdown(report)

with tab3:
    st.markdown("**LangGraph execution trace**")
    for t in trace:
        st.write(f"- {t['node']} ({t['ms']} ms)")
