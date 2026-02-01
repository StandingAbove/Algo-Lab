# agent_engine.py
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv(override=True)

import os
from typing import Any, Dict, Optional, TypedDict

import pandas as pd

from tool import (
    fetch_prices_yfinance,
    compute_technicals,
    compute_kpis,
    monte_carlo_outlook,
    retrieve_context_llamaindex,
)

# LangGraph
from langgraph.graph import StateGraph, END

# Optional Groq
try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    ChatGroq = None  # type: ignore
    SystemMessage = None  # type: ignore
    HumanMessage = None  # type: ignore


class ResearchState(TypedDict, total=False):
    # Inputs
    ticker: str
    lookback_years: int
    horizon_days: int
    n_sims: int
    docs_dir: str

    # Data artifacts
    df: pd.DataFrame
    audit: Dict[str, Any]
    kpis: Dict[str, Any]
    tech: Dict[str, Any]
    outlook: Dict[str, Any]
    retrieved: Dict[str, Any]

    # Output
    report: str
    error: str


def _validate(state: ResearchState) -> ResearchState:
    ticker = (state.get("ticker") or "").strip().upper()
    if not ticker:
        return {"error": "Ticker is empty."}

    lookback_years = int(state.get("lookback_years", 5))
    horizon_days = int(state.get("horizon_days", 60))
    n_sims = int(state.get("n_sims", 3000))
    docs_dir = str(state.get("docs_dir", os.getenv("DOCS_DIR", "./data/docs")))

    if lookback_years < 1 or lookback_years > 20:
        return {"error": "Lookback years must be between 1 and 20."}
    if horizon_days < 5 or horizon_days > 252:
        return {"error": "Horizon must be between 5 and 252 trading days."}
    if n_sims < 200 or n_sims > 20000:
        return {"error": "Simulations must be between 200 and 20000."}

    return {
        "ticker": ticker,
        "lookback_years": lookback_years,
        "horizon_days": horizon_days,
        "n_sims": n_sims,
        "docs_dir": docs_dir,
    }


def _fetch_data(state: ResearchState) -> ResearchState:
    df, audit = fetch_prices_yfinance(
        ticker=state["ticker"],
        lookback_years=state["lookback_years"],
        interval="1d",
    )
    return {"df": df, "audit": audit}


def _compute_numbers(state: ResearchState) -> ResearchState:
    df = compute_technicals(state["df"])
    kpis = compute_kpis(df)

    last = df.iloc[-1]
    tech = {
        "above_ma200": bool(last.get("Above_MA200", False)),
        "ma50_gt_ma200": bool(last.get("MA50_gt_MA200", False)),
        "ma20": None if pd.isna(last.get("MA20")) else float(last["MA20"]),
        "ma50": None if pd.isna(last.get("MA50")) else float(last["MA50"]),
        "ma200": None if pd.isna(last.get("MA200")) else float(last["MA200"]),
    }

    return {"df": df, "kpis": kpis, "tech": tech}


def _simulate_outlook(state: ResearchState) -> ResearchState:
    outlook = monte_carlo_outlook(
        state["df"],
        horizon_days=state["horizon_days"],
        n_sims=state["n_sims"],
        seed=7,
        sample_paths=200,
    )
    return {"outlook": outlook}


def _retrieve_context(state: ResearchState) -> ResearchState:
    """
    LlamaIndex retrieval is optional. If docs_dir is missing/empty or llama-index isn't installed,
    retrieved["enabled"] will be False.
    """
    q = (
        f"{state['ticker']} outlook, risks, catalysts, business context. "
        f"Date range {state['kpis']['start_date']} to {state['kpis']['end_date']}."
    )
    retrieved = retrieve_context_llamaindex(query=q, docs_dir=state["docs_dir"], top_k=4)
    return {"retrieved": retrieved}


def _get_llm() -> Any:
    if ChatGroq is None:
        raise RuntimeError("langchain_groq not installed.")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Missing GROQ_API_KEY.")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(model=model, temperature=0.1)


def _synthesize_report(state: ResearchState) -> ResearchState:
    ticker = state["ticker"]
    k = state["kpis"]
    t = state["tech"]
    o = state["outlook"]
    r = state.get("retrieved", {"enabled": False, "snippets": []})

    # Deterministic fallback if Groq is not configured
    if not os.getenv("GROQ_API_KEY") or ChatGroq is None:
        ctx = ""
        if r.get("enabled") and r.get("snippets"):
            ctx = "\n\n### Retrieved context\n" + "\n".join(f"- {s}" for s in r["snippets"][:4])

        report = (
            f"## `{ticker}` research note\n\n"
            f"### Snapshot\n"
            f"- Date range: **{k['start_date']}** to **{k['end_date']}**\n"
            f"- Last close: **{k['last_close']:.2f}**\n"
            f"- CAGR: **{k['cagr']*100:.2f}%**\n"
            f"- Ann. vol: **{k['ann_vol']*100:.2f}%**\n"
            f"- Max drawdown: **{k['max_drawdown']*100:.2f}%**\n\n"
            f"### Technical context\n"
            f"- Above 200D MA: **{t['above_ma200']}**\n"
            f"- 50D > 200D: **{t['ma50_gt_ma200']}**\n\n"
            f"### Future outlook (simulation)\n"
            f"- Horizon: **{o['horizon_days']}** trading days\n"
            f"- Simulations: **{o['n_sims']}**\n"
            f"- Implied drift (ann): **{o['mu_ann']*100:.2f}%**\n"
            f"- Implied vol (ann): **{o['vol_ann']*100:.2f}%**\n"
            f"- P(final > spot): **{o['prob_finish_up']*100:.1f}%**\n\n"
            f"Note: outlook is a stochastic simulation based on historical returns, not a prediction."
            f"{ctx}"
        )
        return {"report": report}

    llm = _get_llm()

    snippets = r.get("snippets") or []
    context_block = ""
    if r.get("enabled") and snippets:
        context_block = "\n".join(f"- {s}" for s in snippets[:4])

    system = (
        "You are a cautious financial research assistant.\n"
        "Write a concise markdown note.\n"
        "Rules:\n"
        "1) Do not invent numbers.\n"
        "2) Use **bold** only for the provided KPIs/stats.\n"
        "3) Use backticks only for the ticker.\n"
        "4) Under ~260 words.\n"
        "5) Clearly label the future outlook as a simulation, not a prediction.\n"
    )

    user = (
        f"Ticker: {ticker}\n"
        f"Date range: {k['start_date']} to {k['end_date']}\n"
        f"Last close: {k['last_close']:.2f}\n"
        f"CAGR: {k['cagr']*100:.2f}%\n"
        f"Annual return: {k['ann_return']*100:.2f}%\n"
        f"Annual vol: {k['ann_vol']*100:.2f}%\n"
        f"Sharpe (rf=0): {k['sharpe_0rf']:.2f}\n"
        f"Max drawdown: {k['max_drawdown']*100:.2f}%\n"
        f"Above 200D MA: {t['above_ma200']}\n"
        f"50D > 200D: {t['ma50_gt_ma200']}\n"
        f"Outlook horizon (days): {o['horizon_days']}\n"
        f"Outlook simulations: {o['n_sims']}\n"
        f"Implied drift (ann): {o['mu_ann']*100:.2f}%\n"
        f"Implied vol (ann): {o['vol_ann']*100:.2f}%\n"
        f"P(final > spot): {o['prob_finish_up']*100:.1f}%\n\n"
        f"Retrieved context (if any):\n{context_block}\n\n"
        "Write sections: Summary, Risk, Future outlook (simulation), What to watch (3 bullets)."
    )

    msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    report = msg.content if hasattr(msg, "content") else str(msg)
    return {"report": report}


def build_graph():
    g = StateGraph(ResearchState)

    g.add_node("validate", _validate)
    g.add_node("fetch_data", _fetch_data)
    g.add_node("compute_numbers", _compute_numbers)
    g.add_node("simulate_outlook", _simulate_outlook)
    g.add_node("retrieve_context", _retrieve_context)
    g.add_node("synthesize_report", _synthesize_report)

    g.set_entry_point("validate")

    # If validation produced error, stop
    def _route_after_validate(state: ResearchState) -> str:
        return END if state.get("error") else "fetch_data"

    g.add_conditional_edges("validate", _route_after_validate, {END: END, "fetch_data": "fetch_data"})

    g.add_edge("fetch_data", "compute_numbers")
    g.add_edge("compute_numbers", "simulate_outlook")
    g.add_edge("simulate_outlook", "retrieve_context")
    g.add_edge("retrieve_context", "synthesize_report")
    g.add_edge("synthesize_report", END)

    return g.compile()


_GRAPH = None


def run_agentic_research(
    ticker: str,
    lookback_years: int = 5,
    horizon_days: int = 60,
    n_sims: int = 3000,
    docs_dir: Optional[str] = None,
) -> Dict[str, Any]:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()

    init: ResearchState = {
        "ticker": ticker,
        "lookback_years": int(lookback_years),
        "horizon_days": int(horizon_days),
        "n_sims": int(n_sims),
        "docs_dir": docs_dir or os.getenv("DOCS_DIR", "./data/docs"),
    }

    out: ResearchState = _GRAPH.invoke(init)

    if out.get("error"):
        raise RuntimeError(out["error"])

    # Return a clean payload for the UI
    return {
        "ticker": out["ticker"],
        "df": out["df"],
        "kpis": out["kpis"],
        "tech": out["tech"],
        "outlook": out["outlook"],
        "retrieved": out.get("retrieved", {"enabled": False, "snippets": []}),
        "report": out["report"],
        "audit": out.get("audit", {}),
    }
