# Stock Price Predictor

An end-to-end deep learning web app that predicts stock prices using a 2-layer LSTM network trained on historical market data. Enter any stock ticker to train a model in real time and generate a 7-day price forecast with confidence intervals.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red)

## Demo

Enter any stock ticker (AAPL, TSLA, MSFT, etc.) and the app will:
1. Download 10 years of historical data from Yahoo Finance
2. Engineer 5 features from raw price data
3. Train a 2-layer LSTM with early stopping on GPU
4. Display predicted vs actual prices on the test set
5. Generate a 7-day forecast with 90% confidence intervals using Monte Carlo Dropout

## Features

- **Any stock** — works with any ticker available on Yahoo Finance
- **Live training** — model trains fresh for each stock, cached for repeat queries
- **5 engineered features** — closing price, volume, 7-day MA, 30-day MA, daily return
- **Early stopping** — automatically stops training when validation loss plateaus
- **Monte Carlo Dropout** — runs 100 inference passes with dropout enabled to generate uncertainty estimates
- **Forecast table** — day-by-day price forecast with low/high bounds and UP/DOWN signals
- **Key metrics** — RMSE and directional accuracy displayed per stock

## Model Architecture

```
Input (60 days × 5 features)
    → LSTM(hidden=50) → Dropout(0.2)
    → LSTM(hidden=50) → Dropout(0.2)
    → Linear(1)
Output (next day closing price)
```

- **Window size:** 60 trading days
- **Train/test split:** 80/20 (chronological, no shuffling)
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Mean Squared Error
- **Early stopping patience:** 10 epochs

## Results (AAPL)

| Metric | Value |
|---|---|
| RMSE | $36.22 |
| Directional accuracy | 54.4% |
| Training data | 10 years (2015–2025) |
| Early stopping epoch | ~27 |

> Note: the model exhibits a known lag bias — predicted prices track the shape of actual prices well but run approximately $30-50 above actual values due to the model anchoring to historical price levels. Directional accuracy (up/down trend) is a more reliable signal than absolute price predictions.

## Known Limitations

- **Lag bias** — predictions overshoot current prices on strongly trending stocks
- **Single asset** — model trains on one stock at a time, no cross-asset signals
- **No external data** — does not incorporate news sentiment, earnings, or macroeconomic indicators
- **Confidence intervals** — Monte Carlo Dropout captures model uncertainty but does not fully account for compounding forecast error over the 7-day horizon
- **Short history stocks** — tickers with less than 3 years of data (e.g. recent IPOs) will produce less reliable results

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/stock-price-predictor.git
cd stock-price-predictor
pip install -r requirements.txt
streamlit run app.py
```

## Running in Google Colab

The notebook was developed in Google Colab with a free T4 GPU. To run it yourself:

1. Open `stock_predictor.ipynb` in Google Colab
2. Go to Runtime → Change runtime type → T4 GPU
3. Run all cells in order
4. Use the ngrok URL printed at the end to access the web app

## Tech Stack

| Library | Purpose |
|---|---|
| PyTorch | LSTM model definition and training |
| yfinance | Historical stock data download |
| scikit-learn | MinMaxScaler normalization, RMSE calculation |
| pandas / numpy | Data manipulation and feature engineering |
| matplotlib | Charts and visualizations |
| Streamlit | Web app framework |
| pyngrok | Public URL tunneling from Colab |

## Project Structure

```
stock-price-predictor/
├── app.py                  # Streamlit web app
├── stock_predictor.ipynb   # Development notebook
├── requirements.txt        # Python dependencies
└── README.md
```

## What I Learned

- End-to-end ML pipeline from raw data to deployed web app
- Time series preprocessing — windowing, normalization, chronological splitting
- LSTM architecture and why sequential models suit financial data
- Overfitting vs generalization — using dropout and early stopping
- Monte Carlo Dropout for uncertainty estimation
- Feature engineering — moving averages, daily returns
- PyTorch training loop — zero_grad, backward, step
- Streamlit for rapid ML app deployment
