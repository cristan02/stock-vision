import numpy as np
import pandas as pd
import math
import cvxpy as cp
import yfinance as yf
from rest_framework.decorators import api_view
from rest_framework.response import Response
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


# Fetch stock data dynamically
def get_stock_data(ticker):
    stock = yf.download(ticker, period="max")
    return stock

# Prepare dataset for LSTM
def prepare_lstm_data(df, time_step=60):
    df['Close'] = df['Close'].fillna(method='ffill')  # Fill missing values
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['Close']])

    X, y = [], []
    for i in range(len(df_scaled) - time_step):
        X.append(df_scaled[i:i+time_step])
        y.append(df_scaled[i+time_step])

    X, y = np.array(X), np.array(y)
    X_train, X_test = X[:-100], X[-100:]  # Use last 100 days as test
    y_train, y_test = y[:-100], y[-100:]

    return X_train, X_test, y_train, y_test, scaler, df_scaled[-time_step:]

# Define LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(256, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Predict future stock prices
def predict_future_prices(model, last_60_days, scaler, future_days=30):
    future_predictions = []
    future_input = last_60_days.copy().reshape(1, 60, 1)

    for _ in range(future_days):
        predicted_price = model.predict(future_input)[0][0]
        future_predictions.append(predicted_price)

        predicted_price_reshaped = np.array([[predicted_price]])
        future_input = np.append(future_input[:, 1:, :], predicted_price_reshaped.reshape(1, 1, 1), axis=1)

    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten().tolist()

# Django API endpoint
@api_view(['POST'])
def lstm_stock_prediction(request):
    try:
        data = request.data
        ticker = data.get("ticker", "AAPL")
        future_days = int(data.get("days", 30))

        df = get_stock_data(ticker)
        X_train, X_test, y_train, y_test, scaler, last_60_days = prepare_lstm_data(df)

        model = build_lstm_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, batch_size=32, epochs=2, verbose=1)

        # Predict on test data (last 100 rows)
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calculate MAE, MSE, MAPE
        mae = mean_absolute_error(y_test_original, y_pred)
        mse = mean_squared_error(y_test_original, y_pred)
        mape = mean_absolute_percentage_error(y_test_original, y_pred)

        # Format actual vs predicted data for Recharts
        actual_vs_predicted = [
            {
                "date": df.index[-100 + i].strftime("%Y-%m-%d"),
                "Actual": round(float(y_test_original[i][0]), 2),
                "Predicted": round(float(y_pred[i][0]), 2)
            }
            for i in range(len(y_test_original))
        ]

        # Predict future prices
        future_predictions = predict_future_prices(model, last_60_days, scaler, future_days)

        return Response({
            "ticker": ticker,
            "future_predictions": future_predictions,
            "metrics": {
                "MAE": round(mae, 4),
                "MSE": round(mse, 4),
                "MAPE": round(mape, 4)
            },
            "actual_vs_predicted": actual_vs_predicted
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['POST'])
def random_forest_stock_prediction(request):
    try:
        data = request.data
        ticker = data.get("ticker", "AAPL")
        future_days = int(data.get("days", 30))

        df = get_stock_data(ticker)
        df['Close'] = df['Close'].fillna(method='ffill')  # Handle missing values

        # Prepare dataset
        df['Date'] = df.index
        df['Date'] = df['Date'].map(datetime.toordinal)  # Convert dates to ordinal numbers

        X = df[['Date']]
        y = df['Close']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Train Random Forest Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict on test data
        y_pred = model.predict(X_test)

        # Compute metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        # Prepare actual vs predicted data
        actual_vs_predicted = [
            {"date": df.index[len(X_train) + i].strftime("%Y-%m-%d"), "Actual": round(float(y_test.iloc[i]), 2),
             "Predicted": round(float(y_pred[i]), 2)}
            for i in range(len(y_test))
        ]

        # Predict future prices
        last_date = df.index[-1]
        future_dates = [(last_date + timedelta(days=i)).toordinal() for i in range(1, future_days + 1)]
        future_predictions = model.predict(pd.DataFrame(future_dates, columns=['Date'])).tolist()

        return Response({
            "ticker": ticker,
            "future_predictions": future_predictions,
            "metrics": {"MAE": round(mae, 4), "MSE": round(mse, 4), "MAPE": round(mape, 4)},
            "actual_vs_predicted": actual_vs_predicted
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['POST'])
def svm_stock_prediction(request):
    try:
        data = request.data
        ticker = data.get("ticker", "AAPL")
        future_days = int(data.get("days", 30))

        df = get_stock_data(ticker)
        df['Close'] = df['Close'].fillna(method='ffill')  # Handle missing values

        # Prepare dataset
        df['Date'] = df.index
        df['Date'] = df['Date'].map(datetime.toordinal)  # Convert dates to ordinal numbers

        X = df[['Date']]
        y = df['Close']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Train SVM Model
        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        model.fit(X_train, y_train)

        # Predict on test data
        y_pred = model.predict(X_test)

        # Compute metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        # Prepare actual vs predicted data
        actual_vs_predicted = [
            {"date": df.index[len(X_train) + i].strftime("%Y-%m-%d"), "Actual": round(float(y_test.iloc[i]), 2),
             "Predicted": round(float(y_pred[i]), 2)}
            for i in range(len(y_test))
        ]

        # Predict future prices
        last_date = df.index[-1]
        future_dates = [(last_date + timedelta(days=i)).toordinal() for i in range(1, future_days + 1)]
        future_predictions = model.predict(pd.DataFrame(future_dates, columns=['Date'])).tolist()

        return Response({
            "ticker": ticker,
            "future_predictions": future_predictions,
            "metrics": {"MAE": round(mae, 4), "MSE": round(mse, 4), "MAPE": round(mape, 4)},
            "actual_vs_predicted": actual_vs_predicted
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)


@api_view(['POST'])
def validate_tickers(request):
    """
    Validate stock tickers from a space-separated string.
    """
    tickers_string = request.data.get("stocks", "")

    # Ensure input is a non-empty string
    if not isinstance(tickers_string, str) or not tickers_string.strip():
        return Response({"error": "Invalid input. Provide a space-separated string of tickers."}, status=400)

    # Convert string to list and remove extra spaces
    tickers = [ticker.strip().upper() for ticker in tickers_string.split() if ticker.strip()]

    if not tickers:
        return Response({"error": "No valid tickers found in input."}, status=400)

    invalid_tickers = []
    
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        try:
            info = stock.history(period="1d")  # Check if stock data exists
            if info.empty:
                invalid_tickers.append(ticker)
        except:
            invalid_tickers.append(ticker)

    if invalid_tickers:
        return Response({"is_valid": False, "invalid_tickers": invalid_tickers})

    return Response({"is_valid": True})

@api_view(['POST'])
def optimize_portfolio_api(request):
    try:
        # Extract user inputs
        data = request.data  # If only 2 stocks, max_weight = 0.5
        universe_tickers = data.get("tickers", ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOG', 'ZOMATO.NS'])
        investment_budget_usd = data.get("budget", 5000)

        # Fetch USD to INR exchange rate
        exchange_rate_data = yf.download('USDINR=X', period='1d')
        exchange_rate = exchange_rate_data['Close'].iloc[-1] if not exchange_rate_data.empty else 83.0

        # Categorize stocks
        indian_stocks = [ticker for ticker in universe_tickers if ticker.endswith('.NS')]
        us_stocks = [ticker for ticker in universe_tickers if ticker not in indian_stocks]

        # Download historical data
        stock_data = yf.download(universe_tickers, period="5y")['Close']
        returns_all = stock_data.pct_change().dropna()

        # Get latest stock prices
        latest_prices = yf.download(universe_tickers, period='1d', interval='1d')['Close'].iloc[-1].astype(float)
        latest_prices_usd = latest_prices.copy()

        # Convert INR prices to USD
        for ticker in indian_stocks:
            latest_prices_usd[ticker] /= float(exchange_rate)

        # Expected returns & covariance matrix
        exp_returns = returns_all.mean().astype(float)
        cov_matrix = returns_all.cov().astype(float)

        def optimize_portfolio(strategy):
            n = len(universe_tickers)
            if n == 0:
                return None

            w = cp.Variable(n)
            port_variance = cp.quad_form(w, cov_matrix.values)
            constraints = [cp.sum(w) == 1, w >= 0.0001]

            if strategy == "max_sharpe":
                objective = cp.Minimize(port_variance - exp_returns.values @ w)
                constraints.append(exp_returns.values @ w >= 0.001)
            elif strategy == "min_volatility":
                objective = cp.Minimize(port_variance)
            elif strategy == "balanced":
                # 🔹 Dynamic max weight to avoid infeasibility
                max_weight = min(0.4, 1 / max(1, n-1) * 2)  
                objective = cp.Minimize(port_variance - 0.5 * (exp_returns.values @ w))
                constraints.append(w <= max_weight)  
                constraints.append(cp.norm(w, 1) <= 1.01)  
            else:
                return None

            problem = cp.Problem(objective, constraints)
            try:
                problem.solve()
                if w.value is not None:
                    opt_weights = np.maximum(w.value, 0)
                    opt_weights /= np.sum(opt_weights)
                    return {
                        'tickers': universe_tickers,
                        'weights': np.round(opt_weights, 2).tolist(),  # ✅ Rounded to 2 decimal places
                        'expected_return': round(float(exp_returns.values @ opt_weights), 4),  # ✅ Rounded
                        'risk': round(float(np.sqrt(port_variance.value)), 4),  # ✅ Rounded
                        'sharpe_ratio': round(float((exp_returns.values @ opt_weights) / np.sqrt(port_variance.value)), 4) if np.sqrt(port_variance.value) > 0 else 0  # ✅ Rounded
                    }
            except cp.error.SolverError:
                return None

        def create_portfolio_response(title, portfolio):
            if not portfolio:
                return None

            allocated_amounts = investment_budget_usd * np.array(portfolio['weights'])
            shares_allocated, total_spent = [], 0

            for ticker, allocated in zip(portfolio['tickers'], allocated_amounts):
                price_usd = latest_prices_usd.get(ticker, np.nan)
                if pd.isna(price_usd):
                    continue
                num_shares = math.floor(allocated / price_usd)
                spent = num_shares * price_usd
                total_spent += spent
                shares_allocated.append({
                    "symbol": ticker,
                    "shares": num_shares,
                    "price": round(float(price_usd), 2),  # ✅ Rounded
                    "allocated": round(float(allocated), 2),  # ✅ Rounded
                    "spent": round(float(spent), 2),  # ✅ Rounded
                })

            return {
                "name": title,
                "stocks": portfolio['tickers'],
                "optimized_weights": portfolio['weights'],
                "expected_daily_return": portfolio['expected_return'],
                "portfolio_risk": portfolio['risk'],
                "sharpe_ratio": portfolio['sharpe_ratio'],
                "share_allocation": shares_allocated,
                "total_spent": round(total_spent, 2),  # ✅ Rounded
                "budget": investment_budget_usd
            }

        # Compute portfolios for different strategies
        strategies = ["max_sharpe", "min_volatility", "balanced"]
        portfolios = [create_portfolio_response(strategy.replace('_', ' ').title() + " Portfolio", optimize_portfolio(strategy)) for strategy in strategies]

        return Response([p for p in portfolios if p is not None])

    except Exception as e:
        return Response({"error": str(e)}, status=500)
