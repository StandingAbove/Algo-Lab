# agent_engine.py
from __future__ import annotations

"""
Agent Engine (yfinance + optional Groq)

What this file does:
- Pulls OHLCV history from yfinance
- Computes KPIs (CAGR, vol, Sharpe, drawdown)
- Computes simple technical features (MAs, flags)
- Builds a stochastic "Future outlook" using Monte Carlo (GBM on log returns)
- Optionally generates a short Groq markdown note (if GROQ_API_KEY is set)

Design notes:
- yfinance sometimes returns MultiIndex columns; we normalize to a single-level OHLCV DataFrame.
- The outlook is a simulation, not a prediction.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
import math
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    ChatGroq = None  # type: ignore
    SystemMessage = None  # type: ignore
    HumanMessage = None  # type: ignore


# -----------------------------
# Helpers: data normalization
# -----------------------------

def _normalize_yf_df(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    yfinance can return:
      - Single-level columns: Open/High/Low/Close/Adj Close/Volume
      - MultiIndex columns: (Field, Ticker) when group_by or multi-ticker behavior kicks in

    We always return a clean single-level OHLCV DataFrame indexed by date.
    """
    if raw is None or raw.empty:
        raise RuntimeError(f"No data returned for {ticker}.")

    df = raw.copy()

    if isinstance(df.columns, pd.MultiIndex):
        # Typical shape: (field, ticker)
        lvl1 = df.columns.get_level_values(1)
        lvl0 = df.columns.get_level_values(0)

        if ticker in set(lvl1):
            df = df.xs(ticker, axis=1, level=1, drop_level=True)
        else:
            # If there is only one ticker present, drop the second level
            if len(set(lvl1)) == 1:
                df.columns = lvl0
            else:
                raise RuntimeError(
                    "yfinance returned multiple tickers/levels. "
                    "Use a single ticker symbol."
                )

    df.columns = [str(c) for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns from yfinance: {missing}")

    # Force numeric types
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Close"])
    if df.empty:
        raise RuntimeError(f"{ticker}: Close series is empty after cleaning.")

    return df


# -----------------------------
# Fetching
# -----------------------------

def fetch_prices_yfinance(
    ticker: str,
    lookback_years: int = 5,
    interval: str = "1d",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fetch historical OHLCV.
    """
    end = dt.date.today()
    start = dt.date(end.year - lookback_years, end.month, end.day)

    raw = yf.download(
        tickers=ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="column",
    )

    df = _normalize_yf_df(raw, ticker)

    audit: Dict[str, Any] = {
        "source": "yfinance",
        "ticker": ticker,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "interval": interval,
        "rows": int(df.shape[0]),
        "cols": list(df.columns),
    }
    return df, audit


# -----------------------------
# KPIs + technicals
# -----------------------------

def _max_drawdown(close: pd.Series) -> float:
    peak = close.cummax()
    dd = close / peak - 1.0
    return float(dd.min())


def _cagr(close: pd.Series) -> float:
    if close.shape[0] < 2:
        return 0.0
    years = (close.index[-1] - close.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return float((close.iloc[-1] / close.iloc[0]) ** (1.0 / years) - 1.0)


def compute_technicals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].astype(float)

    out["MA20"] = close.rolling(20).mean()
    out["MA50"] = close.rolling(50).mean()
    out["MA200"] = close.rolling(200).mean()

    # Use Series comparisons to avoid alignment issues
    out["Above_MA200"] = close.gt(out["MA200"])
    out["MA50_gt_MA200"] = out["MA50"].gt(out["MA200"])

    return out


def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["Close"].astype(float)
    rets = close.pct_change().dropna()

    vol_ann = float(rets.std(ddof=1) * math.sqrt(252.0)) if rets.shape[0] > 1 else 0.0
    ret_ann = float(rets.mean() * 252.0) if rets.shape[0] > 0 else 0.0
    sharpe = float(ret_ann / vol_ann) if vol_ann > 0 else 0.0

    return {
        "start_date": df.index[0].date().isoformat(),
        "end_date": df.index[-1].date().isoformat(),
        "last_close": float(close.iloc[-1]),
        "cagr": _cagr(close),
        "ann_vol": vol_ann,
        "ann_return": ret_ann,
        "sharpe_0rf": sharpe,
        "max_drawdown": _max_drawdown(close),
        "days": int(df.shape[0]),
    }


# -----------------------------
# Future outlook: Monte Carlo (GBM)
# -----------------------------

def monte_carlo_outlook(
    df: pd.DataFrame,
    horizon_days: int = 60,
    n_sims: int = 3000,
    seed: int = 7,
    sample_paths: int = 200,
) -> Dict[str, Any]:
    """
    Geometric Brownian Motion simulation on daily log returns.

    Output includes:
    - quantile bands over time (10/25/50/75/90)
    - implied drift/vol (annualized) from historical log returns
    - P(final > spot)
    - a small sample of paths for plotting (kept small to avoid big payload)
    """
    close = df["Close"].astype(float)
    s0 = float(close.iloc[-1])

    log_rets = np.log(close / close.shift(1)).dropna()
    if log_rets.shape[0] < 60:
        raise RuntimeError("Not enough data for outlook. Increase lookback.")

    mu = float(log_rets.mean())           # daily drift
    sigma = float(log_rets.std(ddof=1))   # daily vol

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n_sims, horizon_days))

    # Log increments: (mu - 0.5*sigma^2) + sigma*z
    increments = (mu - 0.5 * sigma * sigma) + sigma * z
    log_paths = np.cumsum(increments, axis=1)
    paths = s0 * np.exp(log_paths)

    # Bands
    q10 = np.quantile(paths, 0.10, axis=0)
    q25 = np.quantile(paths, 0.25, axis=0)
    q50 = np.quantile(paths, 0.50, axis=0)
    q75 = np.quantile(paths, 0.75, axis=0)
    q90 = np.quantile(paths, 0.90, axis=0)

    end_prices = paths[:, -1]
    prob_up = float(np.mean(end_prices > s0))

    ann_mu = float(mu * 252.0)
    ann_vol = float(sigma * math.sqrt(252.0))

    sp = int(min(sample_paths, n_sims))
    return {
        "s0": s0,
        "mu_daily": mu,
        "sigma_daily": sigma,
        "mu_ann": ann_mu,
        "vol_ann": ann_vol,
        "horizon_days": int(horizon_days),
        "n_sims": int(n_sims),
        "prob_finish_up": prob_up,
        "bands": {
            "q10": q10.tolist(),
            "q25": q25.tolist(),
            "q50": q50.tolist(),
            "q75": q75.tolist(),
            "q90": q90.tolist(),
        },
        "sample_paths": paths[:sp, :].tolist(),
    }


# -----------------------------
# Groq report (optional)
# -----------------------------

def _get_llm() -> Any:
    if ChatGroq is None:
        raise RuntimeError("langchain_groq not installed. Install it or unset GROQ_API_KEY.")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Missing GROQ_API_KEY.")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(model=model, temperature=0.1)


def generate_report_markdown(
    ticker: str,
    kpis: Dict[str, Any],
    tech: Dict[str, Any],
    outlook: Dict[str, Any],
) -> str:
    # Fallback if Groq isn't configured
    if not os.getenv("GROQ_API_KEY") or ChatGroq is None:
        return (
            f"## `{ticker}`\n\n"
            f"- Date range: **{kpis['start_date']}** to **{kpis['end_date']}**\n"
            f"- Last close: **{kpis['last_close']:.2f}**\n"
            f"- CAGR: **{kpis['cagr']*100:.2f}%**\n"
            f"- Ann. vol: **{kpis['ann_vol']*100:.2f}%**\n"
            f"- Max drawdown: **{kpis['max_drawdown']*100:.2f}%**\n\n"
            f"### Technical context\n"
            f"- Above 200D MA: **{tech['above_ma200']}**\n"
            f"- 50D > 200D: **{tech['ma50_gt_ma200']}**\n\n"
            f"### Outlook (simulation)\n"
            f"- Horizon: **{outlook['horizon_days']}** trading days\n"
            f"- P(final > spot): **{outlook['prob_finish_up']*100:.1f}%**\n"
        )

    llm = _get_llm()
    system = (
        "You are a cautious financial research assistant.\n"
        "Write a concise markdown note.\n"
        "Rules:\n"
        "1) Do not invent numbers.\n"
        "2) Use **bold** only for the provided KPIs/stats.\n"
        "3) Use backticks only for the ticker.\n"
        "4) Under ~240 words.\n"
        "5) Clearly label the outlook as a simulation, not a prediction.\n"
    )

    user = (
        f"Ticker: {ticker}\n"
        f"Date range: {kpis['start_date']} to {kpis['end_date']}\n"
        f"Last close: {kpis['last_close']:.2f}\n"
        f"CAGR: {kpis['cagr']*100:.2f}%\n"
        f"Annual return: {kpis['ann_return']*100:.2f}%\n"
        f"Annual vol: {kpis['ann_vol']*100:.2f}%\n"
        f"Sharpe (rf=0): {kpis['sharpe_0rf']:.2f}\n"
        f"Max drawdown: {kpis['max_drawdown']*100:.2f}%\n"
        f"Above 200D MA: {tech['above_ma200']}\n"
        f"50D > 200D: {tech['ma50_gt_ma200']}\n"
        f"Outlook horizon (days): {outlook['horizon_days']}\n"
        f"Outlook simulations: {outlook['n_sims']}\n"
        f"Implied drift (ann): {outlook['mu_ann']*100:.2f}%\n"
        f"Implied vol (ann): {outlook['vol_ann']*100:.2f}%\n"
        f"P(final > spot): {outlook['prob_finish_up']*100:.1f}%\n\n"
        "Write sections: Summary, Risk, Future outlook (simulation), What to watch (3 bullets)."
    )

    msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return msg.content if hasattr(msg, "content") else str(msg)


# -----------------------------
# Main entry
# -----------------------------

def run_agentic_research(
    ticker: str,
    lookback_years: int = 5,
    horizon_days: int = 60,
    n_sims: int = 3000,
) -> Dict[str, Any]:
    df, audit = fetch_prices_yfinance(ticker=ticker, lookback_years=lookback_years)
    df = compute_technicals(df)

    kpis = compute_kpis(df)
    last = df.iloc[-1]

    tech_snapshot = {
        "above_ma200": bool(last.get("Above_MA200", False)),
        "ma50_gt_ma200": bool(last.get("MA50_gt_MA200", False)),
        "ma20": None if pd.isna(last.get("MA20")) else float(last["MA20"]),
        "ma50": None if pd.isna(last.get("MA50")) else float(last["MA50"]),
        "ma200": None if pd.isna(last.get("MA200")) else float(last["MA200"]),
    }

    outlook = monte_carlo_outlook(df, horizon_days=horizon_days, n_sims=n_sims, seed=7, sample_paths=200)
    report = generate_report_markdown(ticker=ticker, kpis=kpis, tech=tech_snapshot, outlook=outlook)

    return {
        "ticker": ticker,
        "df": df,
        "kpis": kpis,
        "tech": tech_snapshot,
        "outlook": outlook,
        "report": report,
        "audit": audit,
    }
