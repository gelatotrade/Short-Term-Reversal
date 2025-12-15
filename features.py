"""
Feature Engineering - Fokus auf Reversal-Signale
"""

import pandas as pd
import numpy as np
import config


def create_features(df):
    """Erstellt alle Features für eine Aktie."""
    df = df.copy()
    
    # Returns
    for d in [1, 2, 3, 5, 10, 20]:
        df[f'ret_{d}d'] = df['Close'].pct_change(d)
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    
    # Moving Averages
    df['ma5'] = df['Close'].rolling(5).mean()
    df['ma20'] = df['Close'].rolling(20).mean()
    df['ma50'] = df['Close'].rolling(50).mean()
    df['price_vs_ma5'] = df['Close'] / df['ma5'] - 1
    df['price_vs_ma20'] = df['Close'] / df['ma20'] - 1
    df['ma_ratio'] = df['ma5'] / df['ma20']
    
    # Volatilität
    df['volatility'] = df['ret_1d'].rolling(10).std()
    df['volatility_20'] = df['ret_1d'].rolling(20).std()
    
    # Reversal Signale
    df['reversal'] = df['ret_5d']
    df['reversal_z'] = (df['reversal'] - df['reversal'].rolling(60).mean()) / df['reversal'].rolling(60).std()
    
    # Bollinger Bands
    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['bb_pos'] = (df['Close'] - bb_mid) / (2 * bb_std)
    
    # Stochastik
    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['stoch'] = 100 * (df['Close'] - low14) / (high14 - low14)
    
    # Volume
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # Momentum
    df['momentum'] = df['Close'] / df['Close'].shift(10) - 1
    
    # Price Patterns
    df['body'] = abs(df['Close'] - df['Open']) / df['Open']
    df['range'] = (df['High'] - df['Low']) / df['Close']
    
    # Konsekutive Down-Days
    df['down'] = (df['Close'] < df['Close'].shift(1)).astype(int)
    df['down_streak'] = df['down'].rolling(5).sum()
    
    # Target
    df['future_ret'] = df['Close'].shift(-config.HOLDING_PERIOD) / df['Close'] - 1
    df['target'] = (df['future_ret'] > 0.005).astype(int)
    
    return df


def get_features():
    """Liste der Feature-Spalten."""
    return [
        'ret_1d', 'ret_2d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d',
        'rsi', 'price_vs_ma5', 'price_vs_ma20', 'ma_ratio',
        'volatility', 'volatility_20',
        'reversal', 'reversal_z', 'bb_pos', 'stoch',
        'vol_ratio', 'momentum', 'body', 'range', 'down_streak'
    ]


def prepare_data(stock_data):
    """Bereitet Daten für ML vor."""
    all_data = []
    features = get_features()
    
    for ticker, df in stock_data.items():
        df = create_features(df)
        df['ticker'] = ticker
        all_data.append(df)
    
    combined = pd.concat(all_data)
    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined = combined.dropna(subset=features + ['target'])
    
    return combined, features
