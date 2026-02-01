# Quant Research Agent

A lightweight quantitative research dashboard built with **Streamlit**, **yfinance**, and optional **Groq LLM** support.

The app pulls historical market data, computes key risk/return metrics, visualizes price action with technical indicators, and generates a **stochastic future outlook** using Monte Carlo simulation. It is designed to look and feel like a real buy-side research tool, not a toy demo.

---

## Features

### Market Data

* Historical OHLCV data via `yfinance`
* Robust handling of ticker formats and data edge cases
* Clean normalization of price data

### Analytics

* CAGR
* Annualized return and volatility
* Sharpe ratio (rf = 0)
* Maximum drawdown
* Moving averages (20 / 50 / 200 day)
* Trend flags (price vs MA200, MA50 vs MA200)

### Visualization

* Interactive **candlestick** or **close-line** chart
* Volume chart
* Overlayed moving averages
* Clean, dashboard-style layout

### Future Outlook (Stochastic)

* Monte Carlo simulation using **Geometric Brownian Motion**
* Configurable horizon and number of simulations
* Median path and uncertainty bands (10–90, 25–75 percentiles)
* Probability of finishing above current price
* Clearly labeled as a **simulation**, not a prediction

### Research Notes (Optional)

* Short, structured markdown report generated via **Groq**
* Falls back to a deterministic summary if no API key is provided

---

## Project Structure

```
.
├── app.py               # Streamlit UI
├── agent_engine.py      # Data, analytics, simulation, reporting
├── requirements.txt     # Python dependencies
├── .env                 # Local environment variables (not committed)
└── README.md
```

---

## Installation

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Minimal required packages:

```text
streamlit
yfinance
pandas
numpy
plotly
python-dotenv
```

Optional (for LLM-generated reports):

```text
langchain-groq
langchain-core
```

---

## Environment Variables

Create a `.env` file in the project root.

```env
# Optional: only needed for LLM-generated reports
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

Do **not** commit `.env` to GitHub.

---

## Running the App

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

---

## How the “Prediction” Works

The **Future Outlook** section uses a Monte Carlo simulation based on historical **log returns**:

* Drift and volatility are estimated from historical data
* Thousands of future price paths are simulated
* Quantile bands summarize the distribution of outcomes

This is a **probabilistic scenario analysis**, not a forecast or price target. It is meant to support intuition around risk and uncertainty, not to provide trading signals.

---

## Design Philosophy

* Simple models, clearly labeled
* No hidden assumptions
* No fabricated precision
* Emphasis on clarity, risk, and uncertainty

This makes the project suitable for:

* Quant / research portfolios
* Internship or recruiting demos
* Personal experimentation with financial modeling

---

## Possible Extensions

* Volatility modeling (e.g., GARCH-based simulations)
* Benchmark overlays
* Factor or return decomposition
* Strategy backtesting
* Fundamental or macro overlays

---

## Disclaimer

This project is for **educational and research purposes only**.
It is not investment advice.


