# Crypto AI Trading Signal Bot

A Streamlit app that scans crypto coins, computes technical indicators, and
gives a **BUY / SELL / HOLD** verdict with a suggested **target price**,
**stop-loss**, **risk:reward ratio**, and a **suggested holding period**.

This is an **analysis** tool. It does **not** place trades or hold exchange
API keys — you stay in control of every trade.

## How it works

- **Data**: CoinGecko's free public API (no key needed) — top coins by market
  cap plus daily OHLC candles.
- **Indicators**: RSI(14), MACD(12,26,9), SMA20/SMA50 trend, ATR(14) for
  volatility.
- **Signal engine** (`signal_engine.py`): a rules-based scoring model that
  weighs each indicator and combines them into one verdict + confidence
  score. It's labeled "AI-style" because it mimics how a discretionary
  trader stacks confluence, but it is not a trained ML model — you can
  swap in your own model or plug in an LLM call for narrative reasoning if
  you want (see "Extending" below).
- **Target / stop-loss**: derived from ATR (volatility-based), not fixed
  percentages, so they adapt to how volatile each coin currently is.

## Setup

```bash
git clone <your-repo-url>
cd crypto-ai-trading-bot
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Deploying so you don't run it locally every day

- **Streamlit Community Cloud** (free): push this folder to a GitHub repo,
  go to share.streamlit.io, connect the repo, deploy. You get a permanent
  URL you can open from your phone — add that URL to your phone's home
  screen like a normal app.
- Any VPS / Render / Railway also works — just run `streamlit run app.py
  --server.port $PORT`.

## Extending

- **Add an LLM reasoning layer**: pass the computed indicators into an
  Anthropic API call and ask it to write a short plain-English rationale
  for the verdict — keep the actual BUY/SELL/HOLD decision rules-based so
  it stays deterministic and auditable.
- **More indicators**: add Bollinger Bands, volume profile, or on-chain
  data in `indicators.py` / `data_fetch.py`.
- **Alerts**: add a Telegram/Discord webhook call when a coin flips to
  BUY/SELL during a scan.
- **Backtesting**: before trusting any signal with real money, backtest
  `signal_engine.py`'s logic against historical data.

## Important disclaimer

This tool is for education and idea-generation only. It is not financial
advice, not a guarantee of future performance, and crypto markets carry
substantial risk of loss. Position size and risk according to your own
judgment (or a licensed advisor's), not solely on this app's output.
