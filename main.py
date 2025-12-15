#!/usr/bin/env python3
"""
ML Trading Bot v3.1 - Short-Term Reversal
==========================================
Mit erweiterten Metriken und 25 Jahren Backtest.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime

import config
from data_fetcher import fetch_data
from features import create_features, prepare_data, get_features
from model import TradingModel, print_metrics
from backtest import Backtester, print_results, plot_results


def main():
    """Hauptprogramm."""
    show_plot = '--plot' in sys.argv
    
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║          ML TRADING BOT v3.1 - REVERSAL                ║
    ║                                                        ║
    ║   • 25 Jahre Backtest                                  ║
    ║   • Ensemble ML (Random Forest + Gradient Boosting)    ║
    ║   • Erweiterte Metriken (Sharpe, Sortino, Calmar)      ║
    ║   • Stop-Loss / Take-Profit / Risk Management          ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    start = datetime.now()
    print(f"Start: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Daten laden
    print("\n" + "="*60)
    print("📊 DATEN LADEN")
    print("="*60)
    stock_data = fetch_data()
    print(f"\n✓ {len(stock_data)} Aktien geladen")
    
    # 2. Features erstellen
    print("\n" + "="*60)
    print("🔧 FEATURES ERSTELLEN")
    print("="*60)
    combined, features = prepare_data(stock_data)
    print(f"✓ {len(combined)} Datenpunkte")
    print(f"✓ {len(features)} Features")
    
    # 3. Modell trainieren
    print("\n" + "="*60)
    print("🧠 MODELL TRAINIEREN")
    print("="*60)
    model = TradingModel()
    metrics = model.train(combined, features)
    print_metrics(metrics)
    
    # 4. Backtest
    print("\n" + "="*60)
    print("📈 BACKTEST")
    print("="*60)
    backtester = Backtester()
    results = backtester.run(model, combined, features, stock_data)
    print_results(results)
    
    if show_plot:
        plot_results(results, save='backtest.png')
    
    # 5. Aktuelle Signale
    print("\n" + "="*60)
    print("🎯 AKTUELLE SIGNALE")
    print("="*60)
    
    signals = []
    for ticker, df in stock_data.items():
        df = create_features(df)
        if df.empty:
            continue
        
        latest = df.iloc[-1:]
        if latest[features].isna().any().any():
            continue
        
        try:
            pred = model.predict(latest)[0]
            prob = model.predict_proba(latest)[0, 1]
            reversal = latest['reversal'].values[0]
            rsi = latest['rsi'].values[0]
            price = df['Close'].iloc[-1]
            
            if pred == 1 and prob > config.MIN_CONFIDENCE and reversal < 0:
                signals.append({
                    'ticker': ticker,
                    'price': price,
                    'confidence': prob,
                    'reversal': reversal,
                    'rsi': rsi,
                    'stop_loss': price * (1 - config.STOP_LOSS),
                    'take_profit': price * (1 + config.TAKE_PROFIT)
                })
        except:
            continue
    
    signals.sort(key=lambda x: x['confidence'], reverse=True)
    
    if signals:
        print(f"\n{'Ticker':<8} {'Price':>10} {'Conf':>8} {'Rev':>8} {'RSI':>6} {'SL':>10} {'TP':>10}")
        print("-"*70)
        for s in signals[:10]:
            print(f"{s['ticker']:<8} ${s['price']:>8.2f} {s['confidence']:>7.1%} "
                  f"{s['reversal']:>+7.2%} {s['rsi']:>5.0f} "
                  f"${s['stop_loss']:>8.2f} ${s['take_profit']:>8.2f}")
    else:
        print("\n⏸️  Keine Kaufsignale aktuell")
    
    end = datetime.now()
    print(f"\n{'='*60}")
    print(f"✅ Fertig in {(end-start).seconds}s")
    print(f"{'='*60}")
    
    return results


if __name__ == '__main__':
    results = main()
