from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from agent_engine import run_agentic_research

st.set_page_config(page_title="Quant Research Agent", layout="wide")

st.markdown(
    """
    <style>
      .block-container { max-width: 1200px; padding-top: 1rem; }
      [data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.08); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Quant Research Agent")
st.caption("LangGraph workflow · GPT-5 synthesis · yfinance data · optional LlamaIndex / SEC EDGAR context")

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
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close"))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50"))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], name="MA200"))
        fig.update_layout(height=520, title=f"{ticker.upper()} Price + Moving Averages")
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
    fig2.add_trace(go.Scatter(x=x, y=b["q50"], name="Median"))
    fig2.add_trace(go.Scatter(x=x, y=b["q10"], name="10th"))
    fig2.add_trace(go.Scatter(x=x, y=b["q90"], name="90th", fill="tonexty"))
    fig2.update_layout(height=420, title="Simulated price distribution bands")
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.markdown(report)

with tab3:
    st.markdown("**LangGraph execution trace**")
    for t in trace:
        st.write(f"- {t['node']} ({t['ms']} ms)")
