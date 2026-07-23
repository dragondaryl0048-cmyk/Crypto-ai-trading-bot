"""
Scalping-speed signal generator for the Streamlit dashboard.

Uses Binance's free public REST API (no key needed — this only reads
public market data, it never touches your account) to pull 1m/5m/15m
candles, which CoinGecko's free tier doesn't reliably provide.

This module is DISPLAY-ONLY: it never places orders. You decide whether
to act on what it shows.
"""
import requests
import pandas as pd

from indicators import ema, rsi, atr

BINANCE_BASE = "https://api.binance.com/api/v3"
TIMEOUT = 10

INTERVAL_LABELS = {"1m": "1 minute", "5m": "5 minutes", "15m": "15 minutes"}


class ScalpDataError(Exception):
    pass


def coingecko_symbol_to_binance_pair(symbol: str) -> str:
    """e.g. 'BTC' -> 'BTCUSDT'. Assumes a USDT spot pair exists."""
    return f"{symbol.upper()}USDT"


def fetch_klines(binance_symbol: str, interval: str = "1m", limit: int = 100) -> pd.DataFrame:
    try:
        resp = requests.get(
            f"{BINANCE_BASE}/klines",
            params={"symbol": binance_symbol, "interval": interval, "limit": limit},
            timeout=TIMEOUT,
        )
    except requests.exceptions.RequestException as e:
        raise ScalpDataError(f"Could not reach Binance: {e}")

    if resp.status_code != 200:
        raise ScalpDataError(
            f"No live data for {binance_symbol} (status {resp.status_code}). "
            "This coin may not have a USDT pair on Binance."
        )

    raw = resp.json()
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_base", "taker_quote", "ignore",
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["time"] = pd.to_datetime(df["close_time"], unit="ms")
    return df[["time", "open", "high", "low", "close", "volume"]]


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

    # tight, scalp-scale target/stop from ATR on this fast timeframe
    stop_loss = price - 1.0 * atr_now
    target = price + 1.5 * atr_now

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
