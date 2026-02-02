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
        border-right: 1px solid rgba(148, 163, 184, 0.18);
        background: linear-gradient(180deg, rgba(8, 12, 20, 0.98), rgba(6, 10, 18, 0.95));
        backdrop-filter: blur(18px);
      }

      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: rgba(226, 232, 240, 0.78);
      }

      .sidebar-header {
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
        margin-bottom: 1rem;
      }

      .sidebar-title {
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.4px;
        color: #f8fafc;
      }

      .sidebar-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(56, 189, 248, 0.16);
        border: 1px solid rgba(56, 189, 248, 0.45);
        color: #7dd3fc;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        width: fit-content;
      }

      .sidebar-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 0.85rem 0.9rem;
        margin-bottom: 0.85rem;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.35);
      }

      .sidebar-card-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
      }

      .sidebar-divider {
        height: 1px;
        background: rgba(148, 163, 184, 0.2);
        margin: 0.6rem 0;
      }

      [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.35);
      }

      .hero {
        background: linear-gradient(120deg, rgba(15, 118, 210, 0.28), rgba(14, 22, 40, 0.9));
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.35);
      }

      .hero h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2.2rem;
      }

      .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(56, 189, 248, 0.14);
        border: 1px solid rgba(56, 189, 248, 0.5);
        color: #7dd3fc;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        letter-spacing: 0.4px;
      }

      .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.2rem;
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

      [data-testid="stInfo"] {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.25);
        color: #e2e8f0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="badge">RESEARCH PROJECT</div>
      <h1>Quant Research Agent</h1>
      <p>Agentic market intelligence with Monte Carlo forecasts, technical signals, and contextual filings.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

plot_theme = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e2e8f0"),
)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-header">
          <div class="sidebar-chip">TRADING DESK</div>
          <div class="sidebar-title">Research Controls</div>
          <p>Configure the market scope and simulation engine.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-card-title">Market Focus</div>', unsafe_allow_html=True)
    ticker = st.text_input("Ticker", "NVDA")
    lookback = st.selectbox("Lookback (years)", [1, 3, 5, 10], index=2)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    horizon = st.slider("Outlook horizon (days)", 20, 252, 60)
    sims = st.selectbox("Monte Carlo simulations", [1000, 3000, 5000], index=1)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-card-title">Context Layer</div>', unsafe_allow_html=True)
    docs_dir = st.text_input("Local docs dir (optional)", "./data/docs")
    use_sec = st.toggle("If no local docs, pull SEC filing", value=True)
    forms = st.multiselect("Filing types", ["10-K", "10-Q", "S-1"], default=["10-K", "10-Q", "S-1"])
    sec_max_chars = st.slider("SEC text max length", 3000, 20000, 12000, step=1000)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-card-title">Run</div>', unsafe_allow_html=True)
    st.caption("Launch the agentic workflow and generate the dashboard.")
    run = st.button("Run research", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

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
        st.markdown('<div class="section-title">Market trend & moving averages</div>', unsafe_allow_html=True)
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
        st.markdown('<div class="section-title">Context source</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="section-title">Future outlook (Monte Carlo simulation)</div>', unsafe_allow_html=True)
    x = list(range(1, outlook["horizon_days"] + 1))
    b = outlook["bands"]
    sample_paths = outlook.get("sample_paths", [])

    fig2 = go.Figure()
    for path in sample_paths[:40]:
        fig2.add_trace(
            go.Scatter(
                x=x,
                y=path,
                mode="lines",
                line=dict(color="rgba(148, 163, 184, 0.15)", width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )
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

    mc_cols = st.columns(3)
    mc_cols[0].metric("Prob finish up", f"{outlook['prob_finish_up']*100:.1f}%")
    mc_cols[1].metric("Ann drift (μ)", f"{outlook['mu_ann']*100:.2f}%")
    mc_cols[2].metric("Ann vol (σ)", f"{outlook['vol_ann']*100:.2f}%")

with tab2:
    st.markdown(report)

with tab3:
    st.markdown("**LangGraph execution trace**")
    for t in trace:
        st.write(f"- {t['node']} ({t['ms']} ms)")
