"""
Backtesting Engine v3.1 - Erweiterte Metriken
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config


class Backtester:
    """Backtester mit vollständigen Metriken."""
    
    def __init__(self):
        self.capital = config.INITIAL_CAPITAL
        self.position_size = config.POSITION_SIZE
        self.max_positions = config.MAX_POSITIONS
        self.stop_loss = config.STOP_LOSS
        self.take_profit = config.TAKE_PROFIT
        self.costs = config.TRANSACTION_COST
    
    def run(self, model, df, features, stock_data):
        """Führt Backtest durch."""
        split = int(len(df) * config.TRAIN_TEST_SPLIT)
        test = df.iloc[split:].copy()
        
        test['pred'] = model.predict(test)
        test['prob'] = model.predict_proba(test)[:, 1]
        
        capital = config.INITIAL_CAPITAL
        positions = {}
        portfolio = []
        trades = []
        daily_returns = []
        
        dates = test.index.unique()
        prev_value = capital
        
        for i, date in enumerate(dates):
            day = test.loc[[date]] if date in test.index else pd.DataFrame()
            if day.empty:
                continue
            
            to_close = []
            for ticker, pos in positions.items():
                if ticker not in stock_data or date not in stock_data[ticker].index:
                    continue
                
                price = stock_data[ticker].loc[date, 'Close']
                ret = (price - pos['entry']) / pos['entry']
                days = i - pos['day']
                
                close = False
                reason = ''
                
                if ret <= -self.stop_loss:
                    close, reason = True, 'STOP_LOSS'
                    price = pos['entry'] * (1 - self.stop_loss)
                elif ret >= self.take_profit:
                    close, reason = True, 'TAKE_PROFIT'
                    price = pos['entry'] * (1 + self.take_profit)
                elif days >= config.HOLDING_PERIOD:
                    close, reason = True, 'HOLDING'
                
                if close:
                    pnl = (price - pos['entry']) * pos['shares']
                    capital += pos['shares'] * price * (1 - self.costs)
                    trades.append({
                        'ticker': ticker,
                        'entry_date': pos['entry_date'],
                        'exit_date': date,
                        'entry_price': pos['entry'],
                        'exit_price': price,
                        'shares': pos['shares'],
                        'return': ret,
                        'pnl': pnl,
                        'days_held': days,
                        'reason': reason
                    })
                    to_close.append(ticker)
            
            for t in to_close:
                del positions[t]
            
            if len(positions) < self.max_positions:
                buys = day[(day['pred'] == 1) & (day['prob'] > config.MIN_CONFIDENCE)]
                buys = buys[buys['reversal'] < 0]
                buys = buys.sort_values('prob', ascending=False)
                
                for _, row in buys.iterrows():
                    if len(positions) >= self.max_positions:
                        break
                    
                    ticker = row['ticker']
                    if ticker in positions or ticker not in stock_data:
                        continue
                    if date not in stock_data[ticker].index:
                        continue
                    
                    price = stock_data[ticker].loc[date, 'Close']
                    invest = capital * self.position_size
                    shares = int(invest / price)
                    
                    if shares > 0 and shares * price * 1.001 <= capital:
                        capital -= shares * price * (1 + self.costs)
                        positions[ticker] = {
                            'entry': price,
                            'shares': shares,
                            'day': i,
                            'entry_date': date
                        }
            
            value = capital
            for t, p in positions.items():
                if t in stock_data and date in stock_data[t].index:
                    value += p['shares'] * stock_data[t].loc[date, 'Close']
            
            portfolio.append({'date': date, 'value': value})
            
            if prev_value > 0:
                daily_returns.append((value - prev_value) / prev_value)
            prev_value = value
        
        return self._calc_results(portfolio, trades, daily_returns)
    
    def _calc_results(self, portfolio, trades, daily_returns):
        """Berechnet alle Metriken."""
        pv = pd.Series([p['value'] for p in portfolio])
        dates = [p['date'] for p in portfolio]
        dr = pd.Series(daily_returns)
        
        total_ret = (pv.iloc[-1] / config.INITIAL_CAPITAL - 1) if len(pv) > 0 else 0
        trading_days = len(pv)
        years = trading_days / 252
        annual_ret = (1 + total_ret) ** (1/years) - 1 if years > 0 and total_ret > -1 else 0
        
        daily_vol = dr.std() if len(dr) > 0 else 0
        annual_vol = daily_vol * np.sqrt(252)
        
        rf_daily = 0.02 / 252
        excess_returns = dr - rf_daily
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if len(dr) > 0 and excess_returns.std() > 0 else 0
        
        negative_returns = dr[dr < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 0 else 0
        sortino = np.sqrt(252) * (dr.mean() - rf_daily) / downside_std if downside_std > 0 else 0
        
        peak = pv.expanding().max()
        drawdown = (pv - peak) / peak
        max_dd = drawdown.min() if len(drawdown) > 0 else 0
        
        calmar = annual_ret / abs(max_dd) if max_dd < 0 else 0
        
        max_dd_duration = 0
        current_dd = 0
        for d in drawdown:
            if d < 0:
                current_dd += 1
                max_dd_duration = max(max_dd_duration, current_dd)
            else:
                current_dd = 0
        
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        
        total_trades = len(trades)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t['return'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['return'] for t in losses]) if losses else 0
        
        gross_profit = sum(t['pnl'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        kelly = (win_rate - (1 - win_rate) / (avg_win / abs(avg_loss))) if avg_loss != 0 and avg_win != 0 else 0
        
        avg_holding = np.mean([t['days_held'] for t in trades]) if trades else 0
        best_trade = max([t['return'] for t in trades]) if trades else 0
        worst_trade = min([t['return'] for t in trades]) if trades else 0
        
        exit_reasons = {}
        for t in trades:
            r = t['reason']
            exit_reasons[r] = exit_reasons.get(r, 0) + 1
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        for t in trades:
            if t['pnl'] > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        monthly_returns = []
        if len(portfolio) > 0:
            pv_df = pd.DataFrame(portfolio)
            pv_df['date'] = pd.to_datetime(pv_df['date'])
            pv_df.set_index('date', inplace=True)
            monthly = pv_df['value'].resample('ME').last().pct_change().dropna()
            monthly_returns = monthly.tolist()
        
        positive_months = len([m for m in monthly_returns if m > 0])
        negative_months = len([m for m in monthly_returns if m <= 0])
        
        return {
            'total_return': total_ret,
            'annual_return': annual_ret,
            'monthly_returns': monthly_returns,
            'positive_months': positive_months,
            'negative_months': negative_months,
            'daily_volatility': daily_vol,
            'annual_volatility': annual_vol,
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar,
            'max_drawdown': max_dd,
            'max_dd_duration': max_dd_duration,
            'trades': total_trades,
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'kelly': kelly,
            'avg_holding': avg_holding,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'exit_reasons': exit_reasons,
            'portfolio': [p['value'] for p in portfolio],
            'dates': dates,
            'trade_list': trades,
            'daily_returns': daily_returns,
            'trading_days': trading_days,
            'years': years,
            'start_capital': config.INITIAL_CAPITAL,
            'end_capital': pv.iloc[-1] if len(pv) > 0 else config.INITIAL_CAPITAL
        }


def print_results(r):
    """Zeigt ausführliche Backtest-Ergebnisse."""
    print("\n" + "="*65)
    print("               VOLLSTÄNDIGER BACKTEST REPORT")
    print("="*65)
    
    print(f"\n📅 ZEITRAUM")
    print(f"   {'Handelstage:':<30} {r['trading_days']:>12}")
    print(f"   {'Jahre:':<30} {r['years']:>12.2f}")
    
    print(f"\n💰 RETURNS")
    print(f"   {'Total Return:':<30} {r['total_return']:>+12.2%}")
    print(f"   {'Annualisierter Return:':<30} {r['annual_return']:>+12.2%}")
    print(f"   {'Positive Monate:':<30} {r['positive_months']:>12}")
    print(f"   {'Negative Monate:':<30} {r['negative_months']:>12}")
    
    print(f"\n📊 RISIKO-METRIKEN")
    print(f"   {'Tägliche Volatilität:':<30} {r['daily_volatility']:>12.4f}")
    print(f"   {'Jährliche Volatilität:':<30} {r['annual_volatility']:>12.2%}")
    print(f"   {'Sharpe Ratio:':<30} {r['sharpe']:>+12.2f}")
    print(f"   {'Sortino Ratio:':<30} {r['sortino']:>+12.2f}")
    print(f"   {'Calmar Ratio:':<30} {r['calmar']:>+12.2f}")
    print(f"   {'Max Drawdown:':<30} {r['max_drawdown']:>12.2%}")
    print(f"   {'Max DD Dauer (Tage):':<30} {r['max_dd_duration']:>12}")
    
    print(f"\n📈 TRADE-STATISTIKEN")
    print(f"   {'Gesamt Trades:':<30} {r['trades']:>12}")
    print(f"   {'Gewinner:':<30} {r['wins']:>12}")
    print(f"   {'Verlierer:':<30} {r['losses']:>12}")
    print(f"   {'Win Rate:':<30} {r['win_rate']:>12.1%}")
    print(f"   {'Durchschn. Gewinn:':<30} {r['avg_win']:>+12.2%}")
    print(f"   {'Durchschn. Verlust:':<30} {r['avg_loss']:>+12.2%}")
    print(f"   {'Bester Trade:':<30} {r['best_trade']:>+12.2%}")
    print(f"   {'Schlechtester Trade:':<30} {r['worst_trade']:>+12.2%}")
    print(f"   {'Profit Factor:':<30} {r['profit_factor']:>12.2f}")
    print(f"   {'Expectancy:':<30} {r['expectancy']:>+12.4f}")
    print(f"   {'Kelly Criterion:':<30} {r['kelly']:>12.2%}")
    print(f"   {'Ø Haltedauer (Tage):':<30} {r['avg_holding']:>12.1f}")
    print(f"   {'Max konsek. Gewinne:':<30} {r['max_consecutive_wins']:>12}")
    print(f"   {'Max konsek. Verluste:':<30} {r['max_consecutive_losses']:>12}")
    
    print(f"\n🚪 EXIT-GRÜNDE")
    for reason, count in r['exit_reasons'].items():
        pct = count / r['trades'] * 100 if r['trades'] > 0 else 0
        print(f"   {reason:<30} {count:>6} ({pct:>5.1f}%)")
    
    print(f"\n💵 KAPITAL")
    print(f"   {'Startkapital:':<30} ${r['start_capital']:>12,.2f}")
    print(f"   {'Endkapital:':<30} ${r['end_capital']:>12,.2f}")
    print(f"   {'Profit/Loss:':<30} ${r['end_capital'] - r['start_capital']:>+12,.2f}")
    
    print("="*65)


def plot_results(r, save=None):
    """Visualisiert Ergebnisse."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('ML Trading Bot v3.1 - Backtest Results', fontsize=14, fontweight='bold')
    
    ax = axes[0, 0]
    ax.plot(r['portfolio'], color='#2E86AB', linewidth=1.5)
    ax.axhline(config.INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.7)
    ax.fill_between(range(len(r['portfolio'])), r['portfolio'], config.INITIAL_CAPITAL,
                    where=[v > config.INITIAL_CAPITAL for v in r['portfolio']], 
                    color='green', alpha=0.2)
    ax.fill_between(range(len(r['portfolio'])), r['portfolio'], config.INITIAL_CAPITAL,
                    where=[v <= config.INITIAL_CAPITAL for v in r['portfolio']], 
                    color='red', alpha=0.2)
    ax.set_title('Portfolio Value')
    ax.set_ylabel('$')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    pv = pd.Series(r['portfolio'])
    dd = (pv - pv.expanding().max()) / pv.expanding().max() * 100
    ax.fill_between(range(len(dd)), dd, 0, color='#E94F37', alpha=0.6)
    ax.set_title(f'Drawdown (Max: {r["max_drawdown"]:.1%})')
    ax.set_ylabel('%')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    if r['monthly_returns']:
        colors = ['green' if m > 0 else 'red' for m in r['monthly_returns']]
        ax.bar(range(len(r['monthly_returns'])), [m*100 for m in r['monthly_returns']], color=colors, alpha=0.7)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_title('Monthly Returns')
    ax.set_ylabel('%')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    if r['trade_list']:
        rets = [t['return'] * 100 for t in r['trade_list']]
        ax.hist(rets, bins=40, color='#2E86AB', edgecolor='white', alpha=0.7)
        ax.axvline(0, color='red', linewidth=2)
        ax.axvline(np.mean(rets), color='green', linestyle='--', label=f'Avg: {np.mean(rets):.1f}%')
        ax.legend()
    ax.set_title('Trade Returns Distribution')
    ax.set_xlabel('Return %')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    if r['trades'] > 0:
        ax.pie([r['wins'], r['losses']], labels=['Wins', 'Losses'], 
               colors=['#2ECC71', '#E74C3C'], autopct='%1.1f%%', startangle=90)
    ax.set_title(f'Win Rate: {r["win_rate"]:.1%}')
    
    ax = axes[1, 2]
    ax.axis('off')
    metrics_text = f"""
    ╔═══════════════════════════════════╗
    ║        KEY METRICS                ║
    ╠═══════════════════════════════════╣
    ║  Total Return:    {r['total_return']:>+10.2%}   ║
    ║  Annual Return:   {r['annual_return']:>+10.2%}   ║
    ║  Sharpe Ratio:    {r['sharpe']:>+10.2f}   ║
    ║  Sortino Ratio:   {r['sortino']:>+10.2f}   ║
    ║  Calmar Ratio:    {r['calmar']:>+10.2f}   ║
    ║  Max Drawdown:    {r['max_drawdown']:>10.2%}   ║
    ╠═══════════════════════════════════╣
    ║  Trades:          {r['trades']:>10}   ║
    ║  Win Rate:        {r['win_rate']:>10.1%}   ║
    ║  Profit Factor:   {r['profit_factor']:>10.2f}   ║
    ║  Expectancy:      {r['expectancy']:>+10.4f}   ║
    ╠═══════════════════════════════════╣
    ║  Start:       ${r['start_capital']:>12,.0f}   ║
    ║  End:         ${r['end_capital']:>12,.0f}   ║
    ╚═══════════════════════════════════╝
    """
    ax.text(0.05, 0.5, metrics_text, fontsize=10, fontfamily='monospace',
            transform=ax.transAxes, verticalalignment='center')
    
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"\n📊 Chart gespeichert: {save}")
    plt.show()
