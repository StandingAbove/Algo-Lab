from __future__ import annotations

import os
import re
import json
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List


def _flatten_yfinance_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _as_series(x):
    if isinstance(x, pd.DataFrame):
        if x.shape[1] >= 1:
            return x.iloc[:, 0]
    return x


# ---------------------------
# Market data
# ---------------------------
def fetch_prices_yfinance(
    ticker: str,
    lookback_years: int = 5,
    interval: str = "1d",
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    end = datetime.utcnow()
    start = end.replace(year=end.year - lookback_years)

    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    df = _flatten_yfinance_columns(df, ticker).dropna().copy()

    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns from yfinance: {missing}. Got: {list(df.columns)}")

    audit = {
        "source": "yfinance",
        "rows": int(df.shape[0]),
        "start": df.index.min().strftime("%Y-%m-%d"),
        "end": df.index.max().strftime("%Y-%m-%d"),
    }
    return df, audit


def fetch_watchlist_snapshot(tickers: List[str], lookback_days: int = 7) -> pd.DataFrame:
    if not tickers:
        raise ValueError("tickers list is empty")

    data = yf.download(
        tickers,
        period=f"{lookback_days}d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if data is None or data.empty:
        raise RuntimeError("No data returned for watchlist")

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
        volume = data["Volume"] if "Volume" in data else None
    else:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
        volume = data[["Volume"]].rename(columns={"Volume": tickers[0]}) if "Volume" in data else None

    rows = []
    for ticker in close.columns:
        series = close[ticker].dropna()
        if series.empty:
            continue
        last = float(series.iloc[-1])
        prev = float(series.iloc[-2]) if len(series) > 1 else last
        change = last - prev
        pct = (change / prev) if prev else 0.0
        vol = float(volume[ticker].iloc[-1]) if volume is not None and ticker in volume else np.nan
        rows.append(
            {
                "Symbol": ticker,
                "Last": last,
                "Chg": change,
                "Chg %": pct,
                "Vol": vol,
            }
        )

    if not rows:
        raise RuntimeError("No rows available for watchlist snapshot")

    frame = pd.DataFrame(rows)
    frame["Trend"] = np.where(frame["Chg"] >= 0, "Up", "Down")
    return frame.sort_values("Chg %", ascending=False).reset_index(drop=True)


# ---------------------------
# Technical indicators
# ---------------------------
def compute_technicals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    close = _as_series(out["Close"])
    out["Return"] = close.pct_change()

    out["MA20"] = close.rolling(20).mean()
    out["MA50"] = close.rolling(50).mean()
    out["MA200"] = close.rolling(200).mean()

    ma200 = _as_series(out["MA200"])
    ma50 = _as_series(out["MA50"])

    out["Above_MA200"] = close > ma200
    out["MA50_gt_MA200"] = ma50 > ma200

    return out


# ---------------------------
# KPIs
# ---------------------------
def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    close = _as_series(df["Close"])
    rets = _as_series(df["Return"]).dropna()

    days = len(rets)
    ann_factor = 252

    total_return = (close.iloc[-1] / close.iloc[0]) - 1
    cagr = (1 + total_return) ** (ann_factor / max(days, 1)) - 1 if days > 0 else np.nan
    ann_vol = rets.std() * np.sqrt(ann_factor) if days > 1 else np.nan
    ann_return = rets.mean() * ann_factor if days > 1 else np.nan
    sharpe_0rf = (ann_return / ann_vol) if (ann_vol and ann_vol > 0) else np.nan

    cum = (1 + rets).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = float(dd.min()) if len(dd) else np.nan

    return {
        "start_date": df.index.min().strftime("%Y-%m-%d"),
        "end_date": df.index.max().strftime("%Y-%m-%d"),
        "days": int(days),
        "last_close": float(close.iloc[-1]),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "ann_return": float(ann_return),
        "ann_vol": float(ann_vol),
        "sharpe_0rf": float(sharpe_0rf),
        "max_drawdown": float(max_dd),
    }


# ---------------------------
# Monte Carlo outlook
# ---------------------------
def monte_carlo_outlook(
    df: pd.DataFrame,
    horizon_days: int = 60,
    n_sims: int = 3000,
    seed: int = 7,
    sample_paths: int = 200,
) -> Dict[str, Any]:
    np.random.seed(seed)

    close = _as_series(df["Close"])
    rets = _as_series(df["Return"]).dropna()

    mu = float(rets.mean())
    sigma = float(rets.std())

    mu_ann = mu * 252
    vol_ann = sigma * np.sqrt(252)

    spot = float(close.iloc[-1])

    sims = np.zeros((n_sims, horizon_days), dtype=float)
    sims[:, 0] = spot

    for t in range(1, horizon_days):
        z = np.random.normal(size=n_sims)
        sims[:, t] = sims[:, t - 1] * np.exp((mu - 0.5 * sigma**2) + sigma * z)

    terminal = sims[:, -1]
    prob_finish_up = float((terminal > spot).mean())

    bands = {
        "q10": np.percentile(sims, 10, axis=0).tolist(),
        "q25": np.percentile(sims, 25, axis=0).tolist(),
        "q50": np.percentile(sims, 50, axis=0).tolist(),
        "q75": np.percentile(sims, 75, axis=0).tolist(),
        "q90": np.percentile(sims, 90, axis=0).tolist(),
    }

    sample = sims[: min(sample_paths, n_sims), :].tolist()

    return {
        "spot": spot,
        "mu_ann": float(mu_ann),
        "vol_ann": float(vol_ann),
        "horizon_days": int(horizon_days),
        "n_sims": int(n_sims),
        "prob_finish_up": prob_finish_up,
        "bands": bands,
        "sample_paths": sample,
    }


# ---------------------------
# LlamaIndex retrieval (optional, never crashes)
# ---------------------------
def retrieve_context_llamaindex(query: str, docs_dir: str, top_k: int = 4) -> Dict[str, Any]:
    """
    Local (no-embeddings) retrieval over files in docs_dir.

    Notes:
    - Does NOT require OpenAI.
    - Works even when you only have GROQ_API_KEY.
    """
    if not docs_dir or not os.path.isdir(docs_dir):
        return {"enabled": False, "source": "local_docs", "snippets": [], "reason": "docs_dir missing"}

    allowed_ext = {".txt", ".md", ".html"}
    files: List[str] = []
    for name in os.listdir(docs_dir):
        if name.startswith("."):
            continue
        p = os.path.join(docs_dir, name)
        if os.path.isfile(p) and os.path.splitext(name.lower())[1] in allowed_ext:
            files.append(p)

    if not files:
        return {"enabled": False, "source": "local_docs", "snippets": [], "reason": "no supported files in docs_dir"}

    # tokenize query (simple)
    q = re.sub(r"[^a-zA-Z0-9\s]", " ", query).lower()
    q_terms = [t for t in q.split() if len(t) > 2]
    if not q_terms:
        return {"enabled": False, "source": "local_docs", "snippets": [], "reason": "empty query"}

    def clean_text(s: str, ext: str) -> str:
        if ext == ".html":
            s = re.sub(r"(?is)<script.*?>.*?</script>", " ", s)
            s = re.sub(r"(?is)<style.*?>.*?</style>", " ", s)
            s = re.sub(r"(?is)<.*?>", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # score paragraphs across all files
    scored: List[tuple[float, str]] = []
    for fp in files:
        ext = os.path.splitext(fp.lower())[1]
        try:
            raw = open(fp, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            continue

        text = clean_text(raw, ext)
        if not text:
            continue

        chunks = re.split(r"(?:\n\s*\n|\r\n\s*\r\n|(?<=[.!?])\s{2,})", text)
        for ch in chunks:
            ch = ch.strip()
            if len(ch) < 80:
                continue
            lower = ch.lower()
            score = 0.0
            for t in q_terms:
                c = lower.count(t)
                if c:
                    score += c
            if score > 0:
                scored.append((score, ch))

    if not scored:
        return {"enabled": False, "source": "local_docs", "snippets": [], "reason": "no matches"}

    scored.sort(key=lambda x: x[0], reverse=True)
    snippets: List[str] = []
    for _, ch in scored[: max(top_k, 1)]:
        snippets.append(ch[:900] + (" ..." if len(ch) > 900 else ""))

    return {"enabled": True, "source": "local_docs", "docs_dir": docs_dir, "snippets": snippets}



# ---------------------------
# SEC EDGAR fallback (no API key)
# ---------------------------
def _sec_headers() -> Dict[str, str]:
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if not ua:
        # still return something, but SEC may block it
        ua = "Mozilla/5.0 (compatible; ResearchAgent/1.0; contact=missing)"
    return {
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json,text/html,*/*",
        "Connection": "keep-alive",
    }


def _sec_get_json(url: str, timeout: int = 30) -> Any:
    resp = requests.get(url, headers=_sec_headers(), timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"SEC HTTP {resp.status_code} for {url}: {resp.text[:200]}")
    return resp.json()


def _sec_get_text(url: str, timeout: int = 30) -> str:
    resp = requests.get(url, headers=_sec_headers(), timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"SEC HTTP {resp.status_code} for {url}: {resp.text[:200]}")
    return resp.text


def sec_lookup_cik_for_ticker(ticker: str) -> str:
    """
    Uses SEC company tickers mapping:
    https://www.sec.gov/files/company_tickers.json
    """
    ticker = ticker.strip().upper()
    mapping = _sec_get_json("https://www.sec.gov/files/company_tickers.json")

    # mapping is dict keyed by ints-as-strings
    for _, row in mapping.items():
        if str(row.get("ticker", "")).upper() == ticker:
            cik_int = int(row["cik_str"])
            return str(cik_int).zfill(10)

    raise RuntimeError(f"Could not find CIK for ticker {ticker}")


def sec_fetch_latest_filing_text(
    ticker: str,
    forms: List[str] | None = None,
    max_chars: int = 12000,
) -> Dict[str, Any]:
    """
    Pull the most recent filing (10-K/10-Q/S-1 etc) from SEC, return cleaned text snippet.
    """
    if forms is None:
        forms = ["10-K", "10-Q", "S-1"]

    cik = sec_lookup_cik_for_ticker(ticker)
    subs = _sec_get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")

    recent = subs.get("filings", {}).get("recent", {})
    forms_list = recent.get("form", [])
    acc_list = recent.get("accessionNumber", [])
    prim_list = recent.get("primaryDocument", [])
    date_list = recent.get("filingDate", [])

    # find most recent matching form
    pick_idx = None
    for i, f in enumerate(forms_list):
        if f in forms:
            pick_idx = i
            break

    if pick_idx is None:
        return {
            "enabled": False,
            "source": "sec",
            "reason": f"no recent filings matched {forms}",
            "snippets": [],
        }

    accession = acc_list[pick_idx].replace("-", "")
    primary = prim_list[pick_idx]
    filing_date = date_list[pick_idx]
    form = forms_list[pick_idx]

    # archives path uses CIK without leading zeros
    cik_nolead = str(int(cik))
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_nolead}/{accession}/{primary}"

    html = _sec_get_text(url)

    # basic HTML -> text
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + " ..."

    return {
        "enabled": True,
        "source": "sec",
        "ticker": ticker,
        "cik": cik,
        "form": form,
        "filing_date": filing_date,
        "url": url,
        "snippets": [text],
    }

def retrieve_context_llamaindex(query: str, docs_dir: str, top_k: int = 4) -> Dict[str, Any]:
    """
    Local (no-embeddings) retrieval over files in docs_dir.

    Notes:
    - Does NOT require OpenAI.
    - Works even when you only have GROQ_API_KEY.
    """
    if not docs_dir or not os.path.isdir(docs_dir):
        return {"enabled": False, "source": "local_docs", "snippets": [], "reason": "docs_dir missing"}

    allowed_ext = {".txt", ".md", ".html"}
    files: List[str] = []
    for name in os.listdir(docs_dir):
        if name.startswith("."):
            continue
        p = os.path.join(docs_dir, name)
        if os.path.isfile(p) and os.path.splitext(name.lower())[1] in allowed_ext:
            files.append(p)

    if not files:
        return {"enabled": False, "source": "local_docs", "snippets": [], "reason": "no supported files in docs_dir"}

    # tokenize query (simple)
    q = re.sub(r"[^a-zA-Z0-9\s]", " ", query).lower()
    q_terms = [t for t in q.split() if len(t) > 2]
    if not q_terms:
        return {"enabled": False, "source": "local_docs", "snippets": [], "reason": "empty query"}

    def clean_text(s: str, ext: str) -> str:
        if ext == ".html":
            s = re.sub(r"(?is)<script.*?>.*?</script>", " ", s)
            s = re.sub(r"(?is)<style.*?>.*?</style>", " ", s)
            s = re.sub(r"(?is)<.*?>", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # score paragraphs across all files
    scored: List[tuple[float, str]] = []
    for fp in files:
        ext = os.path.splitext(fp.lower())[1]
        try:
            raw = open(fp, "r", encoding="utf-8", errors="ignore").read()
        except Exception:
            continue

        text = clean_text(raw, ext)
        if not text:
            continue

        chunks = re.split(r"(?:\n\s*\n|\r\n\s*\r\n|(?<=[.!?])\s{2,})", text)
        for ch in chunks:
            ch = ch.strip()
            if len(ch) < 80:
                continue
            lower = ch.lower()
            score = 0.0
            for t in q_terms:
                c = lower.count(t)
                if c:
                    score += c
            if score > 0:
                scored.append((score, ch))

    if not scored:
        return {"enabled": False, "source": "local_docs", "snippets": [], "reason": "no matches"}

    scored.sort(key=lambda x: x[0], reverse=True)
    snippets: List[str] = []
    for _, ch in scored[: max(top_k, 1)]:
        snippets.append(ch[:900] + (" ..." if len(ch) > 900 else ""))

    return {"enabled": True, "source": "local_docs", "docs_dir": docs_dir, "snippets": snippets}



def _sec_headers() -> Dict[str, str]:
    ua = os.getenv("SEC_USER_AGENT", "").strip()
    if not ua:
        # still return something, but SEC may block it
        ua = "Mozilla/5.0 (compatible; ResearchAgent/1.0; contact=missing)"
    return {
        "User-Agent": ua,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json,text/html,*/*",
        "Connection": "keep-alive",
    }


def _sec_get_json(url: str, timeout: int = 30) -> Any:
    resp = requests.get(url, headers=_sec_headers(), timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"SEC HTTP {resp.status_code} for {url}: {resp.text[:200]}")
    return resp.json()


def _sec_get_text(url: str, timeout: int = 30) -> str:
    resp = requests.get(url, headers=_sec_headers(), timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"SEC HTTP {resp.status_code} for {url}: {resp.text[:200]}")
    return resp.text


def sec_lookup_cik_for_ticker(ticker: str) -> str:
    """
    Uses SEC company tickers mapping:
    https://www.sec.gov/files/company_tickers.json
    """
    ticker = ticker.strip().upper()
    mapping = _sec_get_json("https://www.sec.gov/files/company_tickers.json")

    # mapping is dict keyed by ints-as-strings
    for _, row in mapping.items():
        if str(row.get("ticker", "")).upper() == ticker:
            cik_int = int(row["cik_str"])
            return str(cik_int).zfill(10)

    raise RuntimeError(f"Could not find CIK for ticker {ticker}")


def sec_fetch_latest_filing_text(
    ticker: str,
    forms: List[str] | None = None,
    max_chars: int = 12000,
) -> Dict[str, Any]:
    """
    Pull the most recent filing (10-K/10-Q/S-1 etc) from SEC, return cleaned text snippet.
    """
    if forms is None:
        forms = ["10-K", "10-Q", "S-1"]

    cik = sec_lookup_cik_for_ticker(ticker)
    subs = _sec_get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")

    recent = subs.get("filings", {}).get("recent", {})
    forms_list = recent.get("form", [])
    acc_list = recent.get("accessionNumber", [])
    prim_list = recent.get("primaryDocument", [])
    date_list = recent.get("filingDate", [])

    # find most recent matching form
    pick_idx = None
    for i, f in enumerate(forms_list):
        if f in forms:
            pick_idx = i
            break

    if pick_idx is None:
        return {
            "enabled": False,
            "source": "sec",
            "reason": f"no recent filings matched {forms}",
            "snippets": [],
        }

    accession = acc_list[pick_idx].replace("-", "")
    primary = prim_list[pick_idx]
    filing_date = date_list[pick_idx]
    form = forms_list[pick_idx]

    # archives path uses CIK without leading zeros
    cik_nolead = str(int(cik))
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_nolead}/{accession}/{primary}"

    html = _sec_get_text(url)

    # basic HTML -> text
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + " ..."

    return {
        "enabled": True,
        "source": "sec",
        "ticker": ticker,
        "cik": cik,
        "form": form,
        "filing_date": filing_date,
        "url": url,
        "snippets": [text],
    }
