"""
Market data access via CoinGecko's free public API (no API key required).
Note: the free tier is rate-limited (~10-30 req/min). Streamlit caching
(see app.py) keeps repeat calls to a minimum.
"""
import time
import requests
import pandas as pd

BASE = "https://api.coingecko.com/api/v3"
TIMEOUT = 15


class RateLimitError(Exception):
    """Raised when CoinGecko's free API is rate-limiting us."""
    pass


def _get(url, params=None, retries=4):
    last_status = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT)
        except requests.exceptions.RequestException as e:
            if attempt < retries:
                time.sleep(2 + attempt * 2)
                continue
            raise RateLimitError(f"Network error reaching CoinGecko: {e}")

        if resp.status_code == 200:
            return resp.json()

        last_status = resp.status_code
        if resp.status_code == 429 and attempt < retries:
            time.sleep(3 + attempt * 3)  # back off longer each retry
            continue
        break

    if last_status == 429:
        raise RateLimitError(
            "CoinGecko's free API is rate-limiting requests right now. "
            "Wait a minute and try again, or reduce the number of coins scanned."
        )
    raise RateLimitError(f"CoinGecko request failed (status {last_status}).")


def get_top_coins(limit: int = 50, vs_currency: str = "usd") -> pd.DataFrame:
    """Top coins by market cap — used to populate the scan list."""
    data = _get(f"{BASE}/coins/markets", {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": limit,
        "page": 1,
        "sparkline": "false",
    })
    df = pd.DataFrame(data)[["id", "symbol", "name", "current_price", "market_cap", "price_change_percentage_24h"]]
    df["symbol"] = df["symbol"].str.upper()
    return df


def get_ohlc(coin_id: str, days: int = 30, vs_currency: str = "usd") -> pd.DataFrame:
    """Daily OHLC candles — needed for ATR (stop-loss / target) and trend calc."""
    data = _get(f"{BASE}/coins/{coin_id}/ohlc", {"vs_currency": vs_currency, "days": days})
    df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close"])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    return df
