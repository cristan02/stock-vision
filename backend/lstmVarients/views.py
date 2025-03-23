import numpy as np
import pandas as pd
import yfinance as yf
from rest_framework.decorators import api_view
from rest_framework.response import Response
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

step = 20

# Fetch stock data
def get_stock_data(ticker, start, end):
    stock = yf.download(ticker, start=start, end=end)
    return stock[['Close']]

# Helper function to fetch stock data
def fetch_stock_data(ticker, years=10):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
    return get_stock_data(ticker, start_date, end_date)

# LSTM Preparation
def prepare_lstm_data(df, time_steps=step):
    data = df['Close'].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    X, y = [], []
    for i in range(len(data_scaled) - time_steps):
        X.append(data_scaled[i:i+time_steps])
        y.append(data_scaled[i+time_steps])
    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    return X_train, X_test, y_train, y_test, scaler

# Predict future prices for LSTM
def predict_future_prices_lstm(model, last_60_days, scaler, future_days=30):
    future_predictions = []
    future_input = last_60_days.copy().reshape(1, 60, 1)

    for _ in range(future_days):
        predicted_price = model.predict(future_input)[0][0]
        future_predictions.append(predicted_price)
        predicted_price_reshaped = np.array([[predicted_price]])
        future_input = np.append(future_input[:, 1:, :], predicted_price_reshaped.reshape(1, 1, 1), axis=1)

    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten().tolist()

# Helper function to compute metrics
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 4), "MSE": round(mse, 4), "MAPE": round(mape, 4), "R²": round(r2, 4)}

# Helper function to prepare actual vs predicted data
def prepare_actual_vs_predicted(dates, y_true, y_pred):
    return [
        {"date": dates[i].strftime("%Y-%m-%d"), "Actual": round(float(y_true[i]), 2),
         "Predicted": round(float(y_pred[i]), 2)}
        for i in range(len(y_true))
    ]

# LSTM API
@api_view(['POST'])
def lstm_tanh_stock_prediction(request):
    try:
        data = request.data
        ticker = data.get("ticker", "AAPL")
        future_days = int(data.get("days", 30))

        # Fetch stock data
        df = fetch_stock_data(ticker)

        # Prepare LSTM data
        X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(df, step)

        # Build and train LSTM model
        def build_lstm_model(input_shape):
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=input_shape,activation='tanh'),
                Dropout(0.2),
                LSTM(units=50, return_sequences=True,activation='tanh'),
                Dropout(0.2),
                LSTM(units=50,activation='tanh'),
                Dropout(0.2),
                Dense(units=1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            return model

        model = build_lstm_model((X_train.shape[1], 1))
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, callbacks=[early_stopping])

        # Predict on test data
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Compute metrics
        metrics = compute_metrics(y_test_original, y_pred)

        # Prepare actual vs predicted data
        actual_vs_predicted = prepare_actual_vs_predicted(df.index[-len(y_test):], y_test_original.flatten(), y_pred.flatten())

        # Predict future prices
        last_60_days = df['Close'].values[-60:]
        last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
        future_predictions = predict_future_prices_lstm(model, last_60_days_scaled, scaler, future_days)

        return Response({
            "ticker": ticker,
            "future_predictions": future_predictions,
            "metrics": metrics,
            "actual_vs_predicted": actual_vs_predicted
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['POST'])
def lstm_relu_stock_prediction(request):
    try:
        data = request.data
        ticker = data.get("ticker", "AAPL")
        future_days = int(data.get("days", 30))

        # Fetch stock data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365 * 15)).strftime('%Y-%m-%d')
        df = get_stock_data(ticker, start_date, end_date)

        # Prepare LSTM data
        X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(df, step)

        # Build LSTM model with ReLU activation
        def build_lstm_relu_model(input_shape):
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=input_shape, activation='relu'),
                Dropout(0.2),
                LSTM(units=50, return_sequences=True, activation='relu'),
                Dropout(0.2),
                LSTM(units=50, activation='relu'),
                Dropout(0.2),
                Dense(units=1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            return model

        model = build_lstm_relu_model((X_train.shape[1], 1))
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, callbacks=[early_stopping])

        # Predict on test data
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Compute metrics
        mae = mean_absolute_error(y_test_original, y_pred)
        mse = mean_squared_error(y_test_original, y_pred)
        mape = mean_absolute_percentage_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)

        # Prepare actual vs predicted data
        actual_vs_predicted = [
            {"date": df.index[-len(y_test) + i].strftime("%Y-%m-%d"), "Actual": round(float(y_test_original[i][0]), 2),
             "Predicted": round(float(y_pred[i][0]), 2)}
            for i in range(len(y_test))
        ]

        # Predict future prices
        last_60_days = df['Close'].values[-60:]
        last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
        future_predictions = predict_future_prices_lstm(model, last_60_days_scaled, scaler, future_days)

        return Response({
            "ticker": ticker,
            "future_predictions": future_predictions,
            "metrics": {"MAE": round(mae, 4), "MSE": round(mse, 4), "MAPE": round(mape, 4), "R²": round(r2, 4)},
            "actual_vs_predicted": actual_vs_predicted
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['POST'])
def lstm_bidirectional_stock_prediction(request):
    try:
        data = request.data
        ticker = data.get("ticker", "AAPL")
        future_days = int(data.get("days", 30))

        # Fetch stock data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365 * 15)).strftime('%Y-%m-%d')
        df = get_stock_data(ticker, start_date, end_date)

        # Prepare LSTM data
        X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(df, step)

        # Build Bidirectional LSTM model
        def build_bidirectional_lstm_model(input_shape):
            model = Sequential([
                Bidirectional(LSTM(units=50, return_sequences=True, activation='tanh'), input_shape=input_shape),
                Dropout(0.2),
                Bidirectional(LSTM(units=50, return_sequences=True, activation='tanh')),
                Dropout(0.2),
                Bidirectional(LSTM(units=50, activation='tanh')),
                Dropout(0.2),
                Dense(units=1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            return model

        model = build_bidirectional_lstm_model((X_train.shape[1], 1))
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, callbacks=[early_stopping])

        # Predict on test data
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Compute metrics
        mae = mean_absolute_error(y_test_original, y_pred)
        mse = mean_squared_error(y_test_original, y_pred)
        mape = mean_absolute_percentage_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)

        # Prepare actual vs predicted data
        actual_vs_predicted = [
            {"date": df.index[-len(y_test) + i].strftime("%Y-%m-%d"), "Actual": round(float(y_test_original[i][0]), 2),
             "Predicted": round(float(y_pred[i][0]), 2)}
            for i in range(len(y_test))
        ]

        # Predict future prices
        last_60_days = df['Close'].values[-60:]
        last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
        future_predictions = predict_future_prices_lstm(model, last_60_days_scaled, scaler, future_days)

        return Response({
            "ticker": ticker,
            "future_predictions": future_predictions,
            "metrics": {"MAE": round(mae, 4), "MSE": round(mse, 4), "MAPE": round(mape, 4), "R²": round(r2, 4)},
            "actual_vs_predicted": actual_vs_predicted
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)
    
@api_view(['POST'])
def lstm_conv_stock_prediction(request):
    try:
        data = request.data
        ticker = data.get("ticker", "AAPL")
        future_days = int(data.get("days", 30))

        # Fetch stock data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365 * 15)).strftime('%Y-%m-%d')
        df = get_stock_data(ticker, start_date, end_date)

        # Prepare LSTM data
        X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(df, step)

        # Build LSTM with Convolutional layers
        def build_lstm_conv_model(input_shape):
            model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
                MaxPooling1D(pool_size=2),
                LSTM(units=50, return_sequences=True, activation='tanh'),
                Dropout(0.2),
                LSTM(units=50, activation='tanh'),
                Dropout(0.2),
                Dense(units=1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            return model

        model = build_lstm_conv_model((X_train.shape[1], 1))
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, callbacks=[early_stopping])

        # Predict on test data
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Compute metrics
        mae = mean_absolute_error(y_test_original, y_pred)
        mse = mean_squared_error(y_test_original, y_pred)
        mape = mean_absolute_percentage_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)

        # Prepare actual vs predicted data
        actual_vs_predicted = [
            {"date": df.index[-len(y_test) + i].strftime("%Y-%m-%d"), "Actual": round(float(y_test_original[i][0]), 2),
             "Predicted": round(float(y_pred[i][0]), 2)}
            for i in range(len(y_test))
        ]

        # Predict future prices
        last_60_days = df['Close'].values[-60:]
        last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
        future_predictions = predict_future_prices_lstm(model, last_60_days_scaled, scaler, future_days)

        return Response({
            "ticker": ticker,
            "future_predictions": future_predictions,
            "metrics": {"MAE": round(mae, 4), "MSE": round(mse, 4), "MAPE": round(mape, 4), "R²": round(r2, 4)},
            "actual_vs_predicted": actual_vs_predicted
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)

