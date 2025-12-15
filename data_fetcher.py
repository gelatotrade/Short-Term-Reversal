"""
Daten-Modul - Yahoo Finance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import config


def fetch_data(tickers=None, years=None):
    """Lädt Aktiendaten."""
    tickers = tickers or config.TICKERS
    years = years or config.LOOKBACK_YEARS
    
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    
    data = {}
    print(f"Lade {len(tickers)} Aktien ({years} Jahre)...")
    
    for ticker in tickers:
        try:
            df = yf.Ticker(ticker).history(start=start, end=end)
            if len(df) > 500:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.index = pd.to_datetime(df.index).tz_localize(None)
                data[ticker] = df
                print(f"  ✓ {ticker}: {len(df)} Tage")
        except Exception as e:
            print(f"  ✗ {ticker}: Fehler")
    
    return data
