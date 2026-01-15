# Advanced Volatility Forecasting for Options Markets: A Comparative Study of GARCH and Machine Learning Models

**Master's Final Project (MSc in Computational Mathematics) · Grade: 100/100 · UCAM**

> **Academic Evaluation:** *"An exceptional research project that stands out for its high methodological rigor and impeccable technical execution. It masterfully addresses one of the most complex problems in quantitative finance: volatility forecasting in options markets. The work excels in its exhaustive statistical validation protocol and clear orientation towards practical applicability in risk management and algorithmic trading. This project constitutes a highest-level contribution to the field of computational mathematics."*

---

## 📊 Executive Summary & Financial Motivation

Accurate volatility forecasting is the **cornerstone of pricing, hedging, and risk management in options trading**. Inaccurate forecasts lead directly to mispriced derivatives, ineffective hedges, and unexpected losses.

This project delivers a **production-grade analytical framework** that rigorously implements and compares traditional stochastic models (GARCH-family) against modern machine learning approaches (LSTM, XGBoost). The objective is to identify the most robust methodology for generating reliable signals for real-world financial decision-making.

**Key Value Proposition:** A systematic, reproducible, and statistically validated pipeline that can serve as the foundation for volatility-driven trading strategies or risk assessment modules.

---

## 🧮 Methodology & Technical Implementation

### 1. Problem Formalization & Data
Volatility forecasting is framed as a **supervised learning problem on financial time series**.
*   **Target Variable:** Future realized volatility (calculated via `[e.g., Parkinson estimator, squared returns]`).
*   **Feature Space:** Includes lagged returns/volatilities, technical indicators (e.g., RSI, ATR), and exogenous signals (e.g., VIX index).
*   **Data Source:** `[e.g., Yahoo Finance for underlying asset, CBOE for VIX]`.
*   **Ticker & Period:** `[e.g., SPY, from 2010-01-01 to 2023-12-31]`.

### 2. Model Architecture & Pipeline
The project implements a comparative pipeline across three model families:

| Model Category | Specific Models | Library | Rationale |
| :--- | :--- | :--- | :--- |
| **Classical Stochastic** | GARCH(1,1), EGARCH, GJR-GARCH | `arch` | Industry benchmark for volatility clustering & leverage effects. |
| **Tree-Based ML** | XGBoost Regressor, Gradient Boosting | `xgboost`, `sklearn` | Captures non-linear relationships without strict distributional assumptions. |
| **Deep Learning** | LSTM (Long Short-Term Memory) | `TensorFlow/Keras` or `PyTorch` | Models complex temporal dependencies in sequential data. |

**Pipeline Steps:**
1.  **Data Preparation & Feature Engineering**
2.  **Temporal Walk-Forward Split** (strict out-of-sample validation)
3.  **Model Training & Hyperparameter Optimization**
4.  **Out-of-Sample Prediction & Statistical Validation**
5.  **Economic Significance Analysis** (via a simple strategy backtest)

### 3. Validation Protocol
- **Primary Metrics:** Mean Squared Error (MSE), Mean Absolute Error (MAE), Diebold-Mariano test.
- **Economic Test:** Simulated P&L of a basic options strategy (e.g., volatility targeting) using the forecasts.

---

## 📈 Key Results & Financial Interpretation

### Model Performance (Out-of-Sample)
| Model | MSE (x10⁻⁴) | MAE | DM Test (vs. GARCH) | Economic Value |
| :--- | :--- | :--- | :--- | :--- |
| **GARCH(1,1) (Benchmark)** | `[Value]` | `[Value]` | — | `[Metric, e.g., Sharpe: 0.xx]` |
| **EGARCH** | `[Value]` | `[Value]` | `[p-value]` | `[Metric]` |
| **XGBoost** | `[Value]` | `[Value]` | `[p-value]` | `[Metric]` |
| **LSTM** | `[Value]` | `[Value]` | `[p-value]` | `[Value]` |

**Interpretation & Business Impact:**
The results indicate that `[e.g., the XGBoost model]` consistently outperformed traditional benchmarks, suggesting that capturing non-linear feature interactions provides a measurable edge. For a trading desk, this could translate into `[a concrete hypothetical improvement, e.g., "a 5-10% reduction in hedging costs" or "improved Sharpe ratio for a volatility strategy"]`.

---

## 🚀 How to Reproduce & Use This Project

### 1. Clone the Repository
```bash
git clone https://github.com/ortuns23/Advanced-Volatility-Forecasting-for-Options-Markets-UCAM-MSc.git
cd Advanced-Volatility-Forecasting-for-Options-Markets-UCAM-MSc
