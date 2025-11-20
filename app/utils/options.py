import numpy as np
import pandas as pd

# ==========================================================
# CALL WALL & PUT WALL
# ==========================================================
def get_call_put_walls(calls: pd.DataFrame, puts: pd.DataFrame):
    """
    Return structured rows for call wall and put wall.
    """

    if calls is None or puts is None:
        return None, None

    if calls.empty or puts.empty:
        return None, None

    call_row = calls.loc[calls["openInterest"].idxmax()]
    put_row = puts.loc[puts["openInterest"].idxmax()]

    return call_row, put_row


# ==========================================================
# GAMMA EXPOSURE
# ==========================================================
def calc_gamma_exposure(calls: pd.DataFrame, puts: pd.DataFrame):
    """
    Simplified gamma exposure proxy:
    GEX = sum(OI * IV) for calls  -  sum(OI * IV) for puts.
    """

    if calls is None or puts is None:
        return np.nan

    if calls.empty or puts.empty:
        return np.nan

    calls = calls.copy()
    puts = puts.copy()

    # Ensure numeric IV and OI exist
    calls["impliedVolatility"] = calls["impliedVolatility"].fillna(0)
    puts["impliedVolatility"] = puts["impliedVolatility"].fillna(0)

    calls["gex"] = calls["openInterest"] * calls["impliedVolatility"]
    puts["gex"] = puts["openInterest"] * puts["impliedVolatility"]

    raw_gex = calls["gex"].sum() - puts["gex"].sum()

    try:
        return float(raw_gex)
    except:
        return np.nan
