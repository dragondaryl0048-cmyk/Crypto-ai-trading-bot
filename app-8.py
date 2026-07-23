"""
Crypto AI Trading Signal Bot — Streamlit app.

Run:
    streamlit run app.py

This is an ANALYSIS tool, not an auto-trading bot: it does not place orders
or hold API keys for any exchange. It reads public price data, computes
indicators, and shows you a suggested verdict, target and stop-loss.
"""
import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from data_fetch import get_top_coins, RateLimitError
from signal_engine import analyze_coin
from scalping import coingecko_symbol_to_binance_pair, fetch_klines, scalp_signal, ScalpDataError, INTERVAL_LABELS, get_higher_timeframe_trend
from hourly_engine import analyze_coin_hourly

st.set_page_config(page_title="Crypto AI Trading Signal Bot", layout="wide", page_icon="📈")

VERDICT_COLOR = {"BUY": "#00D68F", "SELL": "#FF5470", "HOLD": "#FFB000"}


@st.cache_data(ttl=300, show_spinner=False)
def cached_top_coins(limit):
    return get_top_coins(limit=limit)


@st.cache_data(ttl=180, show_spinner=False)
def cached_analysis(coin_id, symbol):
    return analyze_coin(coin_id, symbol)


@st.cache_data(ttl=180, show_spinner=False)
def cached_hourly_analysis(symbol, reference_price=None):
    return analyze_coin_hourly(symbol, reference_price)


def disclaimer():
    st.warning(
        "⚠️ **Educational tool, not financial advice.** Signals come from a rules-based "
        "scoring of public price data (RSI, MACD, moving averages, ATR). Crypto markets are "
        "volatile and this can be wrong. Never risk money you can't afford to lose, and size "
        "positions using your own risk management.",
        icon="⚠️",
    )


def render_scan_table(rows: list[dict]):
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No data yet — click Scan.")
        return
    df["ai_prob_up"] = df["ml"].apply(lambda m: m.get("prob_up") if m and m.get("available") else None)
    display = df[["symbol", "price", "verdict", "confidence", "ai_prob_up", "rsi", "target", "stop_loss", "risk_reward", "holding_period"]].copy()
    display.columns = ["Coin", "Price ($)", "Verdict", "Confidence (%)", "AI Prob. Up (%)", "RSI", "Target ($)", "Stop-loss ($)", "R:R", "Holding period"]

    def color_verdict(val):
        color = VERDICT_COLOR.get(val, "#7A8699")
        return f"color: {color}; font-weight: 700"

    st.dataframe(
        display.style.map(color_verdict, subset=["Verdict"]).format({
            "Price ($)": "{:.4f}", "Target ($)": "{:.4f}", "Stop-loss ($)": "{:.4f}",
        }),
        use_container_width=True, hide_index=True, height=560,
    )


def render_coin_detail(result: dict):
    if "error" in result:
        st.error(result["error"])
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verdict", result["verdict"])
    c2.metric("Confidence", f"{result['confidence']}%")
    c3.metric("Entry", f"${result['entry']:.4f}")
    c4.metric("RSI (14)", result["rsi"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Target", f"${result['target']:.4f}")
    c2.metric("Stop-loss", f"${result['stop_loss']:.4f}")
    c3.metric("Risk:Reward", result["risk_reward"])

    st.caption(f"Suggested holding period: **{result['holding_period']}**  ·  ATR: {result['atr_pct']}% of price")

    ml = result.get("ml", {})
    if ml.get("available"):
        st.subheader("🤖 AI Prediction (local ML model)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Probability of up move tomorrow", f"{ml['prob_up']}%")
        m2.metric("Model training accuracy", f"{ml['train_accuracy']}%")
        m3.metric("Training samples used", ml["n_samples"])
        st.caption(
            "Trained fresh on this coin's own recent price history (RandomForest, "
            "scikit-learn) — not a large language model. Training accuracy reflects "
            "fit on past data only and does not guarantee future accuracy."
        )
    else:
        st.info(f"AI prediction unavailable: {ml.get('reason', 'not enough data')}")

    df = result["history"]
    fig = go.Figure(data=[go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color="#00D68F", decreasing_line_color="#FF5470",
    )])
    fig.add_hline(y=result["target"], line_dash="dot", line_color="#00D68F", annotation_text="target")
    fig.add_hline(y=result["stop_loss"], line_dash="dot", line_color="#FF5470", annotation_text="stop-loss")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark",
                       paper_bgcolor="#0B0E11", plot_bgcolor="#0B0E11")
    st.plotly_chart(fig, use_container_width=True)


def render_hourly_table(rows: list[dict]):
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No data yet — click Scan.")
        return
    display = df[["symbol", "price", "verdict", "confidence", "rsi", "target", "stop_loss", "risk_reward", "atr_pct"]].copy()
    display.columns = ["Coin", "Price ($)", "Verdict", "Confidence (%)", "RSI", "Target ($)", "Stop-loss ($)", "R:R", "ATR (%)"]

    def color_verdict(val):
        color = VERDICT_COLOR.get(val, "#7A8699")
        return f"color: {color}; font-weight: 700"

    st.dataframe(
        display.style.map(color_verdict, subset=["Verdict"]).format({
            "Price ($)": "{:.4f}", "Target ($)": "{:.4f}", "Stop-loss ($)": "{:.4f}",
        }),
        use_container_width=True, hide_index=True, height=560,
    )


SCALP_VERDICT_COLOR = {"BUY": "#00D68F", "SELL/EXIT": "#FF5470"}


@st.fragment(run_every=20)
def render_scalp_view(symbol: str, interval: str, reference_price: float = None):
    pair = coingecko_symbol_to_binance_pair(symbol)
    st.caption(f"🔄 Auto-refreshing every 20s · {INTERVAL_LABELS.get(interval, interval)} candles · "
               f"Coinbase public data (read-only) · Querying pair: **{pair}**")
    try:
        df = fetch_klines(pair, interval=interval, limit=100)
        result = scalp_signal(df)
    except ScalpDataError as e:
        st.error(str(e))
        return

    if not result.get("available"):
        st.info(result.get("reason", "not enough data yet"))
        return

    # Sanity check: compare against CoinGecko's independently-fetched price.
    # Catches wrong-pair mappings or stale/bad data before you trust the signal.
    if reference_price and reference_price > 0:
        diff_pct = abs(result["price"] - reference_price) / reference_price * 100
        if diff_pct > 15:
            st.error(
                f"⚠️ Data mismatch: Coinbase shows **${result['price']:.4f}** for {pair}, but CoinGecko's "
                f"reference price for {symbol} is **${reference_price:.4f}** ({diff_pct:.0f}% apart). "
                "Don't trust this signal — likely a wrong pair mapping or stale data. Try refreshing, "
                "or this coin's Coinbase pair may not be the coin you expect."
            )
            return

    verdict_color = SCALP_VERDICT_COLOR.get(result["verdict"], "#FFB000")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"**Verdict**  \n<span style='color:{verdict_color};font-size:1.3rem;font-weight:700'>{result['verdict']}</span>", unsafe_allow_html=True)
    c2.metric("Price", f"${result['price']:.4f}")
    c3.metric("RSI", result["rsi"])
    c4.metric("EMA9 vs EMA21", "above ↑" if result["ema_fast"] > result["ema_slow"] else "below ↓")
    st.caption(f"Reason: {result['reason']}")

    higher_tf = get_higher_timeframe_trend(pair)
    if higher_tf.get("available") and interval != "1h":
        broader = higher_tf["trend"]
        is_bullish_signal = result["verdict"] in ("BUY", "WATCH (uptrend)")
        is_bearish_signal = result["verdict"] in ("SELL/EXIT", "WATCH (downtrend)")
        conflict = (is_bullish_signal and broader == "DOWN") or (is_bearish_signal and broader == "UP")
        trend_color = "#00D68F" if broader == "UP" else "#FF5470"
        st.markdown(f"**Broader 1h trend:** <span style='color:{trend_color};font-weight:700'>{broader}</span>", unsafe_allow_html=True)
        if conflict:
            st.warning(
                f"⚠️ This {INTERVAL_LABELS.get(interval, interval)} signal is going **against** the broader "
                f"1-hour {broader} trend. Counter-trend fast signals whipsaw (reverse) far more often — "
                "treat this one with extra caution.",
                icon="⚠️",
            )

    c1, c2 = st.columns(2)
    direction_note = " (long-style)" if result["verdict"] in ("BUY", "WATCH (uptrend)") else \
                      " (short-style)" if result["verdict"] in ("SELL/EXIT", "WATCH (downtrend)") else ""
    c1.metric(f"Quick target{direction_note}", f"${result['target']:.4f}")
    c2.metric(f"Quick stop-loss{direction_note}", f"${result['stop_loss']:.4f}")

    hist = result["history"]
    fig = go.Figure(data=[go.Candlestick(
        x=hist["time"], open=hist["open"], high=hist["high"], low=hist["low"], close=hist["close"],
        increasing_line_color="#00D68F", decreasing_line_color="#FF5470",
    )])
    fig.add_hline(y=result["target"], line_dash="dot", line_color="#00D68F", annotation_text="target")
    fig.add_hline(y=result["stop_loss"], line_dash="dot", line_color="#FF5470", annotation_text="stop-loss")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), template="plotly_dark",
                       paper_bgcolor="#0B0E11", plot_bgcolor="#0B0E11")
    st.plotly_chart(fig, use_container_width=True, key=f"scalp_chart_{symbol}_{interval}")


def main():
    st.title("📈 Crypto AI Trading Signal Bot")
    disclaimer()

    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Mode", ["Scan all coins", "Single coin deep-dive", "1-Hour Trade Scan", "Scalping (fast signals)"])
        coin_limit = st.slider("Coins to scan", 10, 100, 50, step=10)
        st.caption("Data: CoinGecko public API (free, no key). Cached 5 min.")

    try:
        coins_df = cached_top_coins(coin_limit)
    except RateLimitError as e:
        st.error(f"🚦 {e}")
        st.stop()

    if mode == "Scan all coins":
        st.subheader(f"Scanning top {coin_limit} coins by market cap")
        st.caption("⏱️ Timeframe: **Daily candles** — each data point is one day's close, ~180 days of history per coin. "
                    "This is a swing/position-style view, not a scalping view (use Scalping mode for that).")
        if st.button("🔄 Run scan", type="primary"):
            progress = st.progress(0, text="Analyzing coins…")
            rows = []
            for i, row in coins_df.iterrows():
                try:
                    result = cached_analysis(row["id"], row["symbol"])
                    if "error" not in result:
                        rows.append(result)
                except Exception:
                    pass
                progress.progress((i + 1) / len(coins_df), text=f"Analyzing {row['symbol']}…")
                time.sleep(1.2)  # be polite to the free API's rate limit
            progress.empty()
            st.session_state["scan_rows"] = rows

        rows = st.session_state.get("scan_rows", [])
        render_scan_table(rows)
        if rows:
            n_buy = sum(1 for r in rows if r["verdict"] == "BUY")
            n_sell = sum(1 for r in rows if r["verdict"] == "SELL")
            st.caption(f"{n_buy} BUY · {n_sell} SELL · {len(rows) - n_buy - n_sell} HOLD out of {len(rows)} scanned")

    elif mode == "Single coin deep-dive":
        st.caption("⏱️ Timeframe: **Daily candles** — each data point is one day's close, ~180 days of history. "
                    "Suited to swing/position holds (see the 'Holding period' suggestion below), not scalping.")
        symbol_map = {f"{r['name']} ({r['symbol']})": r["id"] for _, r in coins_df.iterrows()}
        choice = st.selectbox("Choose a coin", list(symbol_map.keys()))
        coin_id = symbol_map[choice]
        symbol = choice.split("(")[-1].strip(")")
        with st.spinner("Analyzing…"):
            result = cached_analysis(coin_id, symbol)
        render_coin_detail(result)

    elif mode == "1-Hour Trade Scan":
        st.subheader(f"1-hour scan · top {coin_limit} coins")
        st.caption("⏱️ Timeframe: **1-hour candles** (Coinbase data) — sized for trades you plan to hold "
                    "roughly within an hour. Faster-moving than the daily Scan, steadier than Scalping mode.")
        if st.button("🔄 Run 1h scan", type="primary"):
            progress = st.progress(0, text="Analyzing coins…")
            rows = []
            symbols = coins_df["symbol"].tolist()
            symbols_prices = dict(zip(coins_df["symbol"], coins_df["current_price"]))
            for i, sym in enumerate(symbols):
                try:
                    result = cached_hourly_analysis(sym, symbols_prices.get(sym))
                    if "error" not in result:
                        rows.append(result)
                except Exception:
                    pass
                progress.progress((i + 1) / len(symbols), text=f"Analyzing {sym}…")
                time.sleep(0.3)
            progress.empty()
            st.session_state["hourly_rows"] = rows

        rows = st.session_state.get("hourly_rows", [])
        render_hourly_table(rows)
        if rows:
            n_buy = sum(1 for r in rows if r["verdict"] == "BUY")
            n_sell = sum(1 for r in rows if r["verdict"] == "SELL")
            st.caption(f"{n_buy} BUY · {n_sell} SELL · {len(rows) - n_buy - n_sell} HOLD out of {len(rows)} scanned "
                        "(coins without a Coinbase USD pair are skipped)")

    else:  # Scalping (fast signals)
        st.warning(
            "⚡ **Scalping mode** shows fast, short-timeframe signals for you to act on manually. "
            "It does not place any trades. Fast timeframes are noisy — false signals are common, "
            "and fees/slippage matter more at this speed than on longer timeframes.",
            icon="⚡",
        )
        symbols = coins_df["symbol"].tolist()
        c1, c2 = st.columns(2)
        symbol = c1.selectbox("Coin (Coinbase USD pair)", symbols)
        interval = c2.selectbox("Candle interval", list(INTERVAL_LABELS.keys()),
                                 format_func=lambda k: INTERVAL_LABELS[k])
        ref_rows = coins_df[coins_df["symbol"] == symbol]
        reference_price = float(ref_rows["current_price"].iloc[0]) if not ref_rows.empty else None
        render_scalp_view(symbol, interval, reference_price)


if __name__ == "__main__":
    main()
