from __future__ import annotations

import os, time
from typing import TypedDict, Dict, Any, List

from dotenv import load_dotenv
load_dotenv(override=True)

from langgraph.graph import StateGraph, END

from tool import (
    fetch_prices_yfinance,
    compute_technicals,
    compute_kpis,
    monte_carlo_outlook,
    retrieve_context_llamaindex,
    sec_fetch_latest_filing_text,
)

try:
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    SystemMessage = None
    HumanMessage = None

try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None


class ResearchState(TypedDict, total=False):
    ticker: str
    lookback_years: int
    horizon_days: int
    n_sims: int

    docs_dir: str
    use_sec: bool
    sec_forms: List[str]
    sec_max_chars: int

    df: Any
    audit: Dict[str, Any]
    kpis: Dict[str, Any]
    tech: Dict[str, Any]
    outlook: Dict[str, Any]
    retrieved: Dict[str, Any]
    report: str

    trace: List[Dict[str, Any]]
    error: str


def _trace(state: ResearchState, node: str, start: float) -> Dict[str, Any]:
    tr = list(state.get("trace", []))
    tr.append({"node": node, "ms": int((time.perf_counter() - start) * 1000)})
    return {"trace": tr}


def validate_node(state: ResearchState) -> ResearchState:
    t0 = time.perf_counter()
    ticker = state["ticker"].strip().upper()
    if not ticker:
        return {"error": "Ticker is empty."}

    return {
        "ticker": ticker,
        "lookback_years": int(state.get("lookback_years", 5)),
        "horizon_days": int(state.get("horizon_days", 60)),
        "n_sims": int(state.get("n_sims", 3000)),
        "docs_dir": state.get("docs_dir", "./data/docs"),
        "use_sec": bool(state.get("use_sec", True)),
        "sec_forms": list(state.get("sec_forms", ["10-K", "10-Q", "S-1"])),
        "sec_max_chars": int(state.get("sec_max_chars", 12000)),
        **_trace(state, "validate", t0),
    }


def fetch_node(state: ResearchState) -> ResearchState:
    t0 = time.perf_counter()
    df, audit = fetch_prices_yfinance(state["ticker"], state["lookback_years"])
    return {"df": df, "audit": audit, **_trace(state, "fetch_data", t0)}


def numbers_node(state: ResearchState) -> ResearchState:
    t0 = time.perf_counter()
    df = compute_technicals(state["df"])
    kpis = compute_kpis(df)

    last = df.iloc[-1]
    tech = {
        "above_ma200": bool(last["Above_MA200"]),
        "ma50_gt_ma200": bool(last["MA50_gt_MA200"]),
    }

    return {"df": df, "kpis": kpis, "tech": tech, **_trace(state, "compute_numbers", t0)}


def outlook_node(state: ResearchState) -> ResearchState:
    t0 = time.perf_counter()
    outlook = monte_carlo_outlook(
        state["df"],
        horizon_days=state["horizon_days"],
        n_sims=state["n_sims"],
    )
    return {"outlook": outlook, **_trace(state, "simulate_outlook", t0)}


def retrieval_node(state: ResearchState) -> ResearchState:
    t0 = time.perf_counter()

    # 1) Try local docs with LlamaIndex (never crashes)
    retrieved = retrieve_context_llamaindex(
        query=f"{state['ticker']} business risks and catalysts",
        docs_dir=state["docs_dir"],
    )

    # 2) If no docs AND SEC enabled, fall back to SEC filing text
    if not retrieved.get("enabled") and state.get("use_sec", True):
        try:
            retrieved = sec_fetch_latest_filing_text(
                ticker=state["ticker"],
                forms=state.get("sec_forms", ["10-K", "10-Q", "S-1"]),
                max_chars=int(state.get("sec_max_chars", 12000)),
            )
        except Exception as e:
            retrieved = {
                "enabled": False,
                "source": "sec",
                "reason": str(e),
                "snippets": [],
            }

    return {"retrieved": retrieved, **_trace(state, "retrieve_context", t0)}


def synthesis_node(state: ResearchState) -> ResearchState:
    t0 = time.perf_counter()

    openai_key = os.getenv("GPT_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()

    if (not groq_key or ChatGroq is None) and (not openai_key or ChatOpenAI is None):
        report = f"Deterministic report for {state['ticker']} (no LLM key configured)."
        return {"report": report, **_trace(state, "synthesize_report", t0)}

    # Prefer Groq (fast, no OpenAI quota). Fallback to OpenAI if configured.
    if groq_key and ChatGroq is not None:
        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
            temperature=0.1,
            api_key=groq_key,
        )
    elif openai_key and ChatOpenAI is not None:
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-5"),
            temperature=0.1,
            api_key=openai_key,
        )
    else:
        raise RuntimeError(
            "No LLM configured. Set GROQ_API_KEY (recommended) or GPT_API_KEY."
        )

    context = state.get("retrieved", {})
    context_snips = context.get("snippets", [])

    sys = SystemMessage(
        content=(
            "You are a cautious quantitative research assistant.\n"
            "Write a concise markdown research note.\n"
            "Do not hallucinate numbers.\n"
            "If you reference SEC content, label it as extracted text and cite the filing type/date.\n"
            "Label forecasts as simulations (Monte Carlo).\n"
        )
    )

    user = HumanMessage(
        content=f"""
Ticker: {state['ticker']}

KPIs:
{state['kpis']}

Signals:
{state['tech']}

Simulation outlook:
{state['outlook']}

Retrieved context metadata:
{ {k: v for k, v in context.items() if k != "snippets"} }

Retrieved text snippets (may be empty):
{context_snips[:2]}

Write sections:
1) Summary (3-6 bullets)
2) Quant snapshot (KPIs + what MA200/MA50 signal implies)
3) Risks and catalysts (use retrieved context if available)
4) Future outlook (simulation; include probabilities and interval interpretation)
5) What to watch (next 30-90 days)
"""
    )

    report = llm.invoke([sys, user]).content
    return {"report": report, **_trace(state, "synthesize_report", t0)}


def build_graph():
    g = StateGraph(ResearchState)

    g.add_node("validate", validate_node)
    g.add_node("fetch", fetch_node)
    g.add_node("numbers", numbers_node)
    g.add_node("outlook", outlook_node)
    g.add_node("retrieve", retrieval_node)
    g.add_node("synthesize", synthesis_node)

    g.set_entry_point("validate")

    g.add_edge("validate", "fetch")
    g.add_edge("fetch", "numbers")
    g.add_edge("numbers", "outlook")
    g.add_edge("outlook", "retrieve")
    g.add_edge("retrieve", "synthesize")
    g.add_edge("synthesize", END)

    return g.compile()


_GRAPH = build_graph()


def run_agentic_research(
    ticker: str,
    lookback_years: int = 5,
    horizon_days: int = 60,
    n_sims: int = 3000,
    docs_dir: str = "./data/docs",
    use_sec: bool = True,
    sec_forms: List[str] | None = None,
    sec_max_chars: int = 12000,
) -> Dict[str, Any]:
    if sec_forms is None:
        sec_forms = ["10-K", "10-Q", "S-1"]

    init: ResearchState = {
        "ticker": ticker,
        "lookback_years": lookback_years,
        "horizon_days": horizon_days,
        "n_sims": n_sims,
        "docs_dir": docs_dir,
        "use_sec": use_sec,
        "sec_forms": sec_forms,
        "sec_max_chars": sec_max_chars,
        "trace": [],
    }

    out = _GRAPH.invoke(init)

    if out.get("error"):
        raise RuntimeError(out["error"])

    return out
