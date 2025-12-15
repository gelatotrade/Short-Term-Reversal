# ML Trading Bot v3.1 + TradingView Indikator

Short-Term Reversal Trading Bot mit Machine Learning und TradingView Integration.

## 🚀 Features

### Python Bot
- **25 Jahre Backtest-Daten**
- **Ensemble ML** (Random Forest + Gradient Boosting)
- **Vollständige Metriken:**
  - Sharpe Ratio
  - Sortino Ratio
  - Calmar Ratio
  - Max Drawdown
  - Profit Factor
  - Kelly Criterion
  - Expectancy
- **Risk Management** (Stop-Loss, Take-Profit)

### TradingView Indikator
- Gleiche Strategie wie Python Bot
- Live-Signale auf jedem Chart
- Dashboard mit allen Metriken
- Alerts für Buy/Sell/SL/TP

---

## 📦 Installation

### Python Bot

```bash
pip3 install -r requirements.txt
python3 main.py --plot
```

### TradingView Indikator

1. Öffne [TradingView](https://tradingview.com)
2. Klicke auf **Pine Editor** (unten)
3. Lösche den Standard-Code
4. Kopiere den Inhalt von `tradingview_indicator.pine`
5. Klicke **Add to Chart**

---

## 📊 Strategie-Erklärung

### Short-Term Reversal

Die Strategie basiert auf **Mean Reversion**:

1. **Aktie ist gefallen** (5-Tage Return < -2%)
2. **RSI überverkauft** (< 40)
3. **Stochastik überverkauft** (< 30)
4. **Preis unter 20-Tage MA**
5. **Mind. 2 konsekutive Down-Days**

→ Wenn mindestens 3 von 5 Bedingungen erfüllt sind = **KAUFSIGNAL**

### Exit-Regeln

- **Stop-Loss:** -5%
- **Take-Profit:** +8%
- **Max Holding:** 5 Tage

---

## 📈 Erwartete Metriken

| Metrik | Erklärung |
|--------|-----------|
| **Sharpe Ratio** | Risk-adjusted Return (>1 = gut) |
| **Sortino Ratio** | Wie Sharpe, aber nur Downside Risk |
| **Calmar Ratio** | Annual Return / Max Drawdown |
| **Profit Factor** | Gross Profit / Gross Loss (>1.5 = gut) |
| **Expectancy** | Durchschn. Gewinn pro Trade |
| **Kelly Criterion** | Optimale Position Size |

---

## 📁 Dateien

```
ml_bot_v3/
├── config.py                  # Einstellungen
├── data_fetcher.py            # Yahoo Finance Daten
├── features.py                # Feature Engineering
├── model.py                   # ML Modell
├── backtest.py                # Backtesting + Metriken
├── main.py                    # Hauptprogramm
├── tradingview_indicator.pine # TradingView Code
└── requirements.txt           # Dependencies
```

---

## ⚠️ Disclaimer

Nur für Bildungszwecke. Keine Finanzberatung. Handeln Sie verantwortungsvoll.
