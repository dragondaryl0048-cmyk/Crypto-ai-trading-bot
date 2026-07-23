"""
1-hour timeframe analysis — for trades you plan to hold roughly within
an hour. Sits between the fast Scalping mode (1m/5m/15m) and the daily
Scan/Deep-dive modes (swing/position holds).

Reuses the same Coinbase public data source as scalping.py (Binance's
API 451-blocks many cloud datacenters, Coinbase doesn't).
"""
import pandas as pd
from indicators import ema, rsi, macd, atr
from scalping import coingecko_symbol_to_binance_pair, fetch_klines, ScalpDataError


def analyze_coin_hourly(symbol: str) -> dict:
    pair = coingecko_symbol_to_binance_pair(symbol)  # -> Coinbase product id, e.g. BTC-USD
    try:
        df = fetch_klines(pair, interval="1h", limit=100)
    except ScalpDataError as e:
        return {"symbol": symbol, "error": str(e)}

    if len(df) < 30:
        return {"symbol": symbol, "error": "not enough hourly history yet"}

    closes, highs, lows = df["close"], df["high"], df["low"]

    rsi_val = rsi(closes, 14).iloc[-1]
    _, _, hist = macd(closes)
    macd_hist_val = hist.iloc[-1]
    ema9 = ema(closes, 9).iloc[-1]
    ema21 = ema(closes, 21).iloc[-1]
    atr_val = atr(highs, lows, closes, 14).iloc[-1]
    price = closes.iloc[-1]
    atr_pct = (atr_val / price) * 100 if price else 0

    # --- scoring (tuned for ~1h holding period) ---
    score = 0
    if rsi_val < 30:
        score += 2
    elif rsi_val > 70:
        score -= 2
    elif rsi_val < 45:
        score += 0.5
    elif rsi_val > 55:
        score -= 0.5
    score += 1 if macd_hist_val > 0 else -1
    score += 1 if ema9 > ema21 else -1

    confidence = min(100, round(abs(score) / 5 * 100))
    if score >= 2.5:
        verdict = "BUY"
    elif score <= -2.5:
        verdict = "SELL"
    else:
        verdict = "HOLD"

    if verdict == "BUY":
        stop_loss = price - 1.2 * atr_val
        target = price + 2.0 * atr_val
    elif verdict == "SELL":
        stop_loss = price + 1.2 * atr_val
        target = price - 2.0 * atr_val
    else:
        stop_loss = price - 1.2 * atr_val
        target = price + 1.2 * atr_val

    risk = abs(price - stop_loss)
    reward = abs(target - price)
    risk_reward = round(reward / risk, 2) if risk else None

    return {
        "symbol": symbol,
        "price": price,
        "rsi": round(rsi_val, 1),
        "ema9": round(ema9, 4),
        "ema21": round(ema21, 4),
        "atr_pct": round(atr_pct, 2),
        "score": round(score, 2),
        "confidence": confidence,
        "verdict": verdict,
        "target": round(target, 4),
        "stop_loss": round(stop_loss, 4),
        "risk_reward": risk_reward,
        "history": df,
    }
