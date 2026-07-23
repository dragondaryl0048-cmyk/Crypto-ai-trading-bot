"""
Scalping-speed signal generator for the Streamlit dashboard.

Uses Coinbase Exchange's free public REST API (no key needed — this only
reads public market data, it never touches any account) to pull 1m/5m/15m
candles. Binance.com's public API returns HTTP 451 from many cloud
providers' US datacenters (Streamlit Cloud included) due to geo-restriction,
so Coinbase is used here instead — it doesn't block cloud IPs.

This module is DISPLAY-ONLY: it never places orders. You decide whether
to act on what it shows.
"""
import requests
import pandas as pd

from indicators import ema, rsi, atr

COINBASE_BASE = "https://api.exchange.coinbase.com"
TIMEOUT = 10

INTERVAL_LABELS = {"1m": "1 minute", "5m": "5 minutes", "15m": "15 minutes", "1h": "1 hour"}
INTERVAL_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}


class ScalpDataError(Exception):
    pass


def coingecko_symbol_to_binance_pair(symbol: str) -> str:
    """
    Kept name for backward compatibility with app.py's import, but this
    now returns a Coinbase Exchange product id, e.g. 'BTC' -> 'BTC-USD'.
    """
    return f"{symbol.upper()}-USD"


def fetch_klines(product_id: str, interval: str = "1m", limit: int = 100) -> pd.DataFrame:
    granularity = INTERVAL_SECONDS.get(interval, 60)
    try:
        resp = requests.get(
            f"{COINBASE_BASE}/products/{product_id}/candles",
            params={"granularity": granularity},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=TIMEOUT,
        )
    except requests.exceptions.RequestException as e:
        raise ScalpDataError(f"Could not reach Coinbase: {e}")

    if resp.status_code != 200:
        raise ScalpDataError(
            f"No live data for {product_id} (status {resp.status_code}). "
            "This coin may not be listed on Coinbase."
        )

    raw = resp.json()
    if not raw:
        raise ScalpDataError(f"No candle data returned for {product_id}.")

    # Coinbase returns [time, low, high, open, close, volume], newest first
    df = pd.DataFrame(raw, columns=["time", "low", "high", "open", "close", "volume"])
    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df[["time", "open", "high", "low", "close", "volume"]].tail(limit).reset_index(drop=True)


def get_higher_timeframe_trend(product_id: str) -> dict:
    """
    Uses 1-hour candles (EMA9 vs EMA50) to gauge the broader trend, so a
    fast-timeframe signal can be flagged when it's fighting the bigger
    picture — a common cause of false scalping signals.
    """
    try:
        df = fetch_klines(product_id, interval="1h", limit=60)
    except ScalpDataError:
        return {"available": False}

    closes = df["close"]
    if len(closes) < 52:
        return {"available": False}

    ema_fast = ema(closes, 9).iloc[-1]
    ema_slow = ema(closes, 50).iloc[-1]
    trend = "UP" if ema_fast > ema_slow else "DOWN"
    return {"available": True, "trend": trend}


def scalp_signal(df: pd.DataFrame, ema_fast: int = 9, ema_slow: int = 21, rsi_period: int = 14) -> dict:
    closes, highs, lows = df["close"], df["high"], df["low"]

    ema_f = ema(closes, ema_fast)
    ema_s = ema(closes, ema_slow)
    rsi_val = rsi(closes, rsi_period)
    atr_val = atr(highs, lows, closes, 14)

    if len(df) < ema_slow + 2:
        return {"available": False, "reason": "not enough candles yet"}

    fast_now, fast_prev = ema_f.iloc[-1], ema_f.iloc[-2]
    slow_now, slow_prev = ema_s.iloc[-1], ema_s.iloc[-2]
    rsi_now = rsi_val.iloc[-1]
    price = closes.iloc[-1]
    atr_now = atr_val.iloc[-1]

    crossed_up = fast_prev <= slow_prev and fast_now > slow_now
    crossed_down = fast_prev >= slow_prev and fast_now < slow_now

    if crossed_up and 40 <= rsi_now <= 70:
        verdict, reason = "BUY", "EMA bullish cross + RSI in healthy range"
    elif crossed_down or rsi_now >= 78:
        verdict, reason = "SELL/EXIT", "EMA bearish cross or RSI overbought"
    elif fast_now > slow_now and rsi_now > 50:
        verdict, reason = "WATCH (uptrend)", "trend intact, no fresh cross yet"
    elif fast_now < slow_now and rsi_now < 50:
        verdict, reason = "WATCH (downtrend)", "trend intact, no fresh cross yet"
    else:
        verdict, reason = "HOLD", "no clear edge right now"

    bullish = verdict in ("BUY", "WATCH (uptrend)")
    bearish = verdict in ("SELL/EXIT", "WATCH (downtrend)")

    if bullish:
        stop_loss = price - 1.0 * atr_now
        target = price + 1.5 * atr_now
    elif bearish:
        stop_loss = price + 1.0 * atr_now
        target = price - 1.5 * atr_now
    else:  # HOLD — no directional call, show a neutral reference range
        stop_loss = price - 1.0 * atr_now
        target = price + 1.0 * atr_now

    return {
        "available": True,
        "price": price,
        "ema_fast": round(fast_now, 4),
        "ema_slow": round(slow_now, 4),
        "rsi": round(rsi_now, 1),
        "atr": atr_now,
        "verdict": verdict,
        "reason": reason,
        "target": round(target, 4),
        "stop_loss": round(stop_loss, 4),
        "history": df,
    }
