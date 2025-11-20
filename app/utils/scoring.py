import numpy as np
import pandas as pd

# ==========================================================
# CORE TREND SCORE
# ==========================================================
def compute_trend_score(df, vix_value, gex, call_wall_strike, put_wall_strike):
    """
    Combine EMA trend, VIX, gamma, and distance from walls into a unified score.
    Returns (score, trend_direction)
    """

    # Force everything to floats/scalars where possible
    try:
        vix_value = float(vix_value)
    except:
        vix_value = np.nan

    try:
        gex = float(gex)
    except:
        gex = np.nan

    try:
        call_wall_strike = float(call_wall_strike)
    except:
        call_wall_strike = None

    try:
        put_wall_strike = float(put_wall_strike)
    except:
        put_wall_strike = None

    if df is None or df.empty:
        return 0, "No Data"

    price = float(df["Close"].iloc[-1])
    ema_fast = float(df["EMA_8"].iloc[-1])
    ema_slow = float(df["EMA_21"].iloc[-1])

    score = 50

    # EMA Trend
    if ema_fast > ema_slow:
        score += 15
        trend_dir = "Uptrend"
    else:
        score -= 15
        trend_dir = "Downtrend"

    # VIX impact
    if not np.isnan(vix_value):
        if 15 <= vix_value <= 22:
            score += 15
        elif 22 < vix_value <= 28:
            score += 5
        elif vix_value < 12:
            score -= 5
        else:
            score -= 10

    # Gamma impact
    if not np.isnan(gex):
        score += 10 if gex < 0 else -10

    # Distance from walls
    if (
        call_wall_strike is not None
        and put_wall_strike is not None
        and not np.isnan(call_wall_strike)
        and not np.isnan(put_wall_strike)
    ):
        dist_call = abs(price - call_wall_strike) / price
        dist_put = abs(price - put_wall_strike) / price
        dist_call = float(dist_call)
        dist_put = float(dist_put)
        dist = min(dist_call, dist_put)

        if dist < 0.004:   # < 0.4%
            score -= 8
        elif dist > 0.01:  # > 1%
            score += 10

    score = max(0, min(100, score))
    return score, trend_dir


# ==========================================================
# BIAS CLASSIFICATION
# ==========================================================
def classify_bias(score, trend_dir, vix_value):
    """
    Convert trend score + direction + volatility into a human-readable bias.
    """
    try:
        vix_value = float(vix_value)
    except:
        vix_value = np.nan

    if score >= 70:
        if not np.isnan(vix_value) and vix_value > 30:
            return "Risk-Off (Extreme Vol)", "risk"
        return (
            "Long Bias (Trend)" if trend_dir == "Uptrend" else "Short Bias (Trend)",
            "long" if trend_dir == "Uptrend" else "short",
        )

    elif score >= 40:
        return "Neutral / Selective", "neutral"

    else:
        return "Risk-Off / Chop", "risk"


# ==========================================================
# EXPECTED MOVE (from VIX)
# ==========================================================
def compute_expected_move(price: float, vix_value: float):
    """
    Approximate 1-day expected move from VIX:
    VIX ~ annualized 30d vol. Daily vol ≈ VIX / sqrt(252) (in %).
    Returns (move_abs, move_pct)
    """
    try:
        price = float(price)
        vix_value = float(vix_value)
    except:
        return np.nan, np.nan

    if price <= 0 or np.isnan(vix_value):
        return np.nan, np.nan

    daily_vol_pct = (vix_value / 100.0) / np.sqrt(252.0)  # as fraction
    move_abs = price * daily_vol_pct
    move_pct = daily_vol_pct * 100.0
    return move_abs, move_pct


# ==========================================================
# MARKET MODE CLASSIFIER
# ==========================================================
def classify_market_mode(trend_score: int, gamma_regime: str, vix_regime: str):
    """
    Classify the overall market mode using:
    - Trend score
    - Gamma regime (Positive/Negative/Neutral)
    - VIX regime (Very Low / Low / Optimal / Elevated / High)

    Returns (mode_label, description)
    """

    gamma_regime = (gamma_regime or "Unknown").strip()
    vix_regime = (vix_regime or "Unknown").strip()

    # Strong trend environment
    if trend_score >= 75 and gamma_regime == "Negative" and vix_regime in ["Optimal", "Elevated"]:
        return "Trending Day", "Market favors directional trades; pullbacks in trend are high quality."

    # Smooth, range-like environment
    if trend_score < 50 and gamma_regime == "Positive" and vix_regime in ["Very Low", "Low"]:
        return "Range-Bound Day", "Mean reversion and fading extremes likely outperform trend following."

    # High volatility chaos
    if vix_regime == "High":
        return "High-Volatility / Chaotic", "Fast moves, big swings, and larger stops required — or stand aside."

    # Balanced/normal
    if 50 <= trend_score < 75 and gamma_regime in ["Positive", "Neutral"] and vix_regime in ["Optimal", "Elevated"]:
        return "Balanced / Two-Sided", "Both sides playable; context and levels matter more than pure trend."

    # Default catch-all
    return "Mixed / Unclear", "Signals are not strongly aligned; reduce size and be selective."
