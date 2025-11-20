import yfinance as yf
import pandas as pd

def get_mnq_data(period="1d", interval="1m"):
    """
    Download MNQ/NQ futures data.
    """
    df = yf.download("NQ=F", period=period, interval=interval, progress=False)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def get_vix_data(period="1d", interval="1m"):
    """
    Download VIX index data.
    """
    df = yf.download("^VIX", period=period, interval=interval, progress=False)
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def get_options_chain(symbol="QQQ"):
    """
    Returns (calls, puts, expiration_str)
    Uses QQQ because NDX options require paid data.
    """
    ticker = yf.Ticker(symbol)
    expirations = ticker.options

    if not expirations:
        return None, None, None

    nearest = expirations[0]

    try:
        chain = ticker.option_chain(nearest)
    except:
        return None, None, None

    calls = chain.calls.copy()
    puts = chain.puts.copy()

    return calls, puts, nearest
