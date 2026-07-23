"""
Combines technical indicators into a single trade signal:
verdict (BUY / SELL / HOLD), confidence, entry, target, stop-loss,
risk:reward ratio, and a suggested holding period.

This is a rules-based "AI-style" scoring model, not a real machine-learning
model or personalized financial advice — see README for the disclaimer.
"""
import pandas as pd
from indicators import sma, rsi, macd, atr
from data_fetch import get_ohlc


def analyze_coin(coin_id: str, symbol: str, days: int = 60) -> dict:
    df = get_ohlc(coin_id, days=days)
    if len(df) < 30:
        return {"coin_id": coin_id, "symbol": symbol, "error": "not enough price history"}

    closes, highs, lows = df["close"], df["high"], df["low"]

    rsi_val = rsi(closes, 14).iloc[-1]
    macd_line, signal_line, hist = macd(closes)
    macd_hist_val = hist.iloc[-1]
    sma20 = sma(closes, 20).iloc[-1]
    sma50 = sma(closes, min(50, len(closes) - 1)).iloc[-1]
    atr_val = atr(highs, lows, closes, 14).iloc[-1]
    price = closes.iloc[-1]
    atr_pct = (atr_val / price) * 100 if price else 0

    # --- scoring ---
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
    score += 1 if sma20 > sma50 else -1
    score += 1 if price > sma50 else -1

    confidence = min(100, round(abs(score) / 6 * 100))

    if score >= 3:
        verdict = "BUY"
    elif score <= -3:
        verdict = "SELL"
    else:
        verdict = "HOLD"

    # --- target / stop-loss from ATR (volatility-based) ---
    if verdict == "BUY":
        stop_loss = price - 1.5 * atr_val
        target = price + 2.5 * atr_val
    elif verdict == "SELL":
        stop_loss = price + 1.5 * atr_val
        target = price - 2.5 * atr_val
    else:
        stop_loss = price - 1.5 * atr_val
        target = price + 1.5 * atr_val

    risk = abs(price - stop_loss)
    reward = abs(target - price)
    risk_reward = round(reward / risk, 2) if risk else None

    # --- suggested holding horizon, based on volatility ---
    if atr_pct > 6:
        horizon = "Short-term (intraday to ~2 days) — high volatility"
    elif atr_pct > 3:
        horizon = "Swing (roughly 3–7 days)"
    else:
        horizon = "Position (roughly 1–3 weeks)"

    return {
        "coin_id": coin_id,
        "symbol": symbol,
        "price": price,
        "rsi": round(rsi_val, 1),
        "macd_hist": round(macd_hist_val, 4),
        "sma20": round(sma20, 4),
        "sma50": round(sma50, 4),
        "atr": atr_val,
        "atr_pct": round(atr_pct, 2),
        "score": round(score, 2),
        "confidence": confidence,
        "verdict": verdict,
        "entry": price,
        "target": round(target, 4),
        "stop_loss": round(stop_loss, 4),
        "risk_reward": risk_reward,
        "holding_period": horizon,
        "history": df,
    }
