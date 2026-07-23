"""
Lightweight machine-learning predictor — trains a small classifier on a
coin's own recent price history to estimate probability of an up move
tomorrow. Free and local (no API key, no external AI service).

This is genuinely ML (scikit-learn RandomForest), not a large language
model — it learns simple statistical patterns from the coin's own past
price/indicator behavior. Treat its output as one more signal, not a
guarantee: markets are noisy and past patterns don't always repeat.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from indicators import sma, rsi, macd, atr


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    closes, highs, lows = df["close"], df["high"], df["low"]
    feat = pd.DataFrame(index=df.index)
    feat["rsi"] = rsi(closes, 14)
    _, _, hist = macd(closes)
    feat["macd_hist"] = hist
    feat["sma20_gap"] = (closes - sma(closes, 20)) / closes
    feat["sma50_gap"] = (closes - sma(closes, 50)) / closes
    feat["atr_pct"] = atr(highs, lows, closes, 14) / closes
    feat["ret_1d"] = closes.pct_change(1)
    feat["ret_3d"] = closes.pct_change(3)
    return feat


def train_and_predict(df: pd.DataFrame, min_rows: int = 60) -> dict:
    """
    df: daily OHLC history (needs at least ~90 rows, i.e. days=180 fetch,
    for the model to have enough training examples after warm-up periods
    used by the indicators).
    Returns probability the price closes higher tomorrow, plus basic
    model diagnostics (train accuracy) so the user can gauge reliability.
    """
    feat = _build_features(df)
    closes = df["close"]
    label = (closes.shift(-1) > closes).astype(int)  # 1 = next day up

    data = feat.copy()
    data["label"] = label
    data = data.dropna()

    if len(data) < min_rows:
        return {"available": False, "reason": f"only {len(data)} usable rows, need {min_rows}+"}

    X = data.drop(columns=["label"])
    y = data["label"]

    # last row = today's features -> what we want to predict for tomorrow
    latest_X = feat.iloc[[-1]].fillna(feat.mean())

    model = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=5,
        random_state=42, class_weight="balanced",
    )
    model.fit(X, y)

    train_acc = model.score(X, y)
    prob_up = model.predict_proba(latest_X)[0][1]

    return {
        "available": True,
        "prob_up": round(prob_up * 100, 1),
        "train_accuracy": round(train_acc * 100, 1),
        "n_samples": len(data),
    }
