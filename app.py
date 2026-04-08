import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=5, hidden_size=50, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x[:, -1, :])
        x = self.fc(x)
        return x

def create_sequences(X_data, y_data, window_size):
    X, y = [], []
    for i in range(window_size, len(X_data)):
        X.append(X_data[i-window_size:i])
        y.append(y_data[i])
    return np.array(X), np.array(y)

def forecast_7_days(model, last_sequence, close_scaler, n_simulations=100):
    model.train()
    all_forecasts = []
    for sim in range(n_simulations):
        forecast = []
        seq = last_sequence.copy()
        for day in range(7):
            input_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(input_tensor).cpu().numpy()[0][0]
            forecast.append(pred)
            new_row = seq[-1].copy()
            new_row[0] = pred
            seq = np.vstack([seq[1:], new_row])
        all_forecasts.append(forecast)
    all_forecasts = np.array(all_forecasts)
    mean_forecast = all_forecasts.mean(axis=0)
    lower_bound = np.percentile(all_forecasts, 5, axis=0)
    upper_bound = np.percentile(all_forecasts, 95, axis=0)
    mean_prices = close_scaler.inverse_transform(mean_forecast.reshape(-1,1)).flatten()
    lower_prices = close_scaler.inverse_transform(lower_bound.reshape(-1,1)).flatten()
    upper_prices = close_scaler.inverse_transform(upper_bound.reshape(-1,1)).flatten()
    return mean_prices, lower_prices, upper_prices

@st.cache_resource
def train_model(ticker):
    data = yf.download(ticker, start="2015-01-01", end=pd.Timestamp.today().strftime("%Y-%m-%d"))
    df = data["Close"][ticker].to_frame(name="Close")
    df["Volume"] = data["Volume"][ticker]
    df["MA7"] = df["Close"].rolling(window=7).mean()
    df["MA30"] = df["Close"].rolling(window=30).mean()
    df["Return"] = df["Close"].pct_change()
    df = df.dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    close_scaler = MinMaxScaler()
    close_scaled = close_scaler.fit_transform(df[["Close"]])

    X, y = create_sequences(scaled, close_scaled, 60)
    y = y.reshape(-1, 1)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

    model = LSTMModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 100
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    progress = st.progress(0)
    status = st.empty()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_test_t)
            val_loss = criterion(val_predictions, y_test_t)

        progress.progress(min((epoch+1)/epochs, 1.0))
        status.text(f"Epoch {epoch+1} — Loss: {loss.item():.4f} — Val Loss: {val_loss.item():.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= patience:
            status.text(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_state)
    progress.empty()

    model.eval()
    with torch.no_grad():
        predicted = model(X_test_t).cpu().numpy()
        actual = y_test_t.cpu().numpy()

    predicted_prices = close_scaler.inverse_transform(predicted)
    actual_prices = close_scaler.inverse_transform(actual)
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))

    signals = []
    for i in range(1, len(predicted_prices)):
        if predicted_prices[i] > predicted_prices[i-1]:
            signals.append("UP")
        else:
            signals.append("DOWN")
    up_pct = signals.count("UP") / len(signals) * 100

    last_sequence = scaled[-60:]
    mean_prices, lower_prices, upper_prices = forecast_7_days(model, last_sequence, close_scaler)
    current_price = df["Close"].iloc[-1]
    close_scaled_full = close_scaler.transform(df[["Close"]])

    return {
        "predicted_prices": predicted_prices,
        "actual_prices": actual_prices,
        "rmse": rmse,
        "up_pct": up_pct,
        "mean_prices": mean_prices,
        "lower_prices": lower_prices,
        "upper_prices": upper_prices,
        "current_price": current_price,
        "close_scaled_full": close_scaled_full,
        "df": df
    }

st.title("Stock price predictor")
st.write("Enter a stock ticker and click predict to train an LSTM model and forecast the next 7 days.")

ticker = st.sidebar.text_input("Stock ticker", value="AAPL").upper()
predict_button = st.sidebar.button("Predict")

if predict_button:
    with st.spinner(f"Downloading data and training model for {ticker}..."):
        results = train_model(ticker)

    current = results["current_price"]
    forecast_day1 = results["mean_prices"][0]
    direction = "UP" if forecast_day1 > current else "DOWN"
    direction_pct = abs((forecast_day1 - current) / current * 100)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current price", f"${current:.2f}")
    col2.metric("Day 1 forecast", f"${forecast_day1:.2f}", f"{direction} {direction_pct:.1f}%")
    col3.metric("RMSE", f"${results['rmse']:.2f}")
    col4.metric("Directional accuracy", f"{results['up_pct']:.1f}%")

    st.subheader("Predicted vs actual price")
    fig1, ax1 = plt.subplots(figsize=(14, 4))
    ax1.plot(results["actual_prices"], label="Actual price")
    ax1.plot(results["predicted_prices"], label="Predicted price")
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Price (USD)")
    ax1.legend()
    st.pyplot(fig1)

    st.subheader("7-day forecast")
    last_30 = close_scaler = MinMaxScaler()
    last_30_prices = results["df"]["Close"].iloc[-30:].values
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.plot(range(-30, 0), last_30_prices, label="Historical price", color="blue")
    ax2.plot(range(0, 7), results["mean_prices"], label="Forecast", color="orange", linewidth=2)
    ax2.fill_between(range(0, 7), results["lower_prices"], results["upper_prices"],
                     alpha=0.3, color="orange", label="90% confidence band")
    ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Days (0 = today)")
    ax2.set_ylabel("Price (USD)")
    ax2.legend()
    st.pyplot(fig2)

    st.subheader("7-day forecast table")
    forecast_df = pd.DataFrame({
        "Day": [f"Day {i+1}" for i in range(7)],
        "Forecast": [f"${p:.2f}" for p in results["mean_prices"]],
        "Low (5%)": [f"${p:.2f}" for p in results["lower_prices"]],
        "High (95%)": [f"${p:.2f}" for p in results["upper_prices"]],
        "Signal": ["UP" if results["mean_prices"][i] > results["mean_prices"][i-1] 
                   else "DOWN" for i in range(7)]
    })
    st.dataframe(forecast_df, use_container_width=True)
