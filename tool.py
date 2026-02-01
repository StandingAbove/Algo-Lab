# tool.py
from __future__ import annotations

import os
import math
import datetime as dt
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# yfinance utilities
# -----------------------------

def normalize_yf_df(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    yfinance can return a MultiIndex columns DataFrame.
    This normalizes to a single-level OHLCV DataFrame.
    """
    if raw is None or raw.empty:
        raise RuntimeError(f"No data returned for {ticker}.")

    df = raw.copy()

    if isinstance(df.columns, pd.MultiIndex):
        lvl1 = df.columns.get_level_values(1)
        lvl0 = df.columns.get_level_values(0)

        if ticker in set(lvl1):
            df = df.xs(ticker, axis=1, level=1, drop_level=True)
        else:
            if len(set(lvl1)) == 1:
                df.columns = lvl0
            else:
                raise RuntimeError(
                    "yfinance returned multiple tickers/levels. Use a single ticker symbol."
                )

    df.columns = [str(c) for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns from yfinance: {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Close"])
    if df.empty:
        raise RuntimeError(f"{ticker}: Close series is empty after cleaning.")

    return df


def fetch_prices_yfinance(
    ticker: str,
    lookback_years: int = 5,
    interval: str = "1d",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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

    df = normalize_yf_df(raw, ticker)

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
# Analytics
# -----------------------------

def compute_technicals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].astype(float)

    out["MA20"] = close.rolling(20).mean()
    out["MA50"] = close.rolling(50).mean()
    out["MA200"] = close.rolling(200).mean()

    out["Above_MA200"] = close.gt(out["MA200"])
    out["MA50_gt_MA200"] = out["MA50"].gt(out["MA200"])
    return out


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
    close = df["Close"].astype(float)
    s0 = float(close.iloc[-1])

    log_rets = np.log(close / close.shift(1)).dropna()
    if log_rets.shape[0] < 60:
        raise RuntimeError("Not enough data for outlook. Increase lookback.")

    mu = float(log_rets.mean())
    sigma = float(log_rets.std(ddof=1))

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n_sims, horizon_days))
    increments = (mu - 0.5 * sigma * sigma) + sigma * z
    log_paths = np.cumsum(increments, axis=1)
    paths = s0 * np.exp(log_paths)

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
# LlamaIndex retrieval (optional)
# -----------------------------

def retrieve_context_llamaindex(
    query: str,
    docs_dir: str = "./data/docs",
    top_k: int = 4,
) -> Dict[str, Any]:
    """
    Builds a small local index from docs_dir and retrieves top_k snippets.
    If docs_dir doesn't exist or is empty, returns empty context.

    Files supported depend on installed LlamaIndex readers (pdf may require extras).
    """
    if not os.path.isdir(docs_dir):
        return {"enabled": False, "docs_dir": docs_dir, "snippets": []}

    # Lazy import so app runs even without llama-index installed
    try:
        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
    except Exception:
        return {"enabled": False, "docs_dir": docs_dir, "snippets": []}

    files = []
    for root, _, names in os.walk(docs_dir):
        for n in names:
            if n.lower().endswith((".txt", ".md", ".pdf")):
                files.append(os.path.join(root, n))

    if not files:
        return {"enabled": False, "docs_dir": docs_dir, "snippets": []}

    # Read + index
    docs = SimpleDirectoryReader(docs_dir).load_data()
    index = VectorStoreIndex.from_documents(docs)

    qe = index.as_query_engine(similarity_top_k=top_k)
    resp = qe.query(query)

    # Response formatting varies by version; keep it simple
    text = str(resp).strip()
    snippets: List[str] = []
    if text:
        # Split into manageable chunks for display
        for chunk in text.split("\n"):
            c = chunk.strip()
            if c:
                snippets.append(c)
            if len(snippets) >= top_k:
                break

    return {"enabled": True, "docs_dir": docs_dir, "snippets": snippets}
