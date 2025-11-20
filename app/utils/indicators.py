import numpy as np
import pandas as pd

# ==========================================================
# EMA CALCULATION
# ==========================================================
def add_emas(df):
    """
    Add EMA_8 and EMA_21 to price dataframe.
    """
    if df.empty:
        return df

    df["EMA_8"] = df["Close"].ewm(span=8).mean()
    df["EMA_21"] = df["Close"].ewm(span=21).mean()
    return df


# ==========================================================
# ATR CALCULATION (Volatility)
# ==========================================================
def add_atr(df, period: int = 14):
    """
    Add ATR (Average True Range) column as ATR_<period>.
    Uses standard Wilder-style TR: max(high-low, |high-prev_close|, |low-prev_close|).
    """
    if df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
        colname = f"ATR_{period}"
        if colname not in df.columns:
            df[colname] = np.nan
        return df

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()

    df[f"ATR_{period}"] = atr
    return df


# ==========================================================
# VIX CLASSIFICATION
# ==========================================================
def classify_vix(vix_value):
    """
    Return (regime_string, explanation_text)
    Accepts floats or Series.
    """
    try:
        vix_value = float(vix_value)
    except:
        return "Unknown", "No data"

    if np.isnan(vix_value):
        return "Unknown", "No data"

    if vix_value < 12:
        return "Very Low", "Choppy, slow market"
    elif 12 <= vix_value < 15:
        return "Low", "Range / slower trends"
    elif 15 <= vix_value <= 22:
        return "Optimal", "Clean price action zone"
    elif 22 < vix_value <= 28:
        return "Elevated", "Volatile but tradable"
    else:
        return "High", "Chaotic movement — caution"


# ==========================================================
# GAMMA CLASSIFICATION
# ==========================================================
def classify_gamma(gex):
    """
    Return (regime_string, explanation_text)
    """
    try:
        gex = float(gex)
    except:
        return "Unknown", "No data"

    if np.isnan(gex):
        return "Unknown", "No data"

    if gex > 0:
        return "Positive", "Mean-reversion, range-bound behavior"
    elif gex < 0:
        return "Negative", "Trend-friendly, expanded moves"
    return "Neutral", "Balanced"
