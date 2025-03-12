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
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


@api_view(['POST'])
def validate_tickers(request):
    """
    Validate stock tickers from a space-separated string and return last day's closing price and date for valid tickers.
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
    valid_tickers = {}

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        try:
            # Fetch historical data for the last available day
            history = stock.history(period="1d")
            if history.empty:
                invalid_tickers.append(ticker)
            else:
                # Get the last day's closing price and date
                last_close_price = history['Close'].iloc[-1]
                last_date = history.index[-1].strftime("%Y-%m-%d")  # Format date as string
                valid_tickers[ticker] = {
                    "last_close_price": round(last_close_price, 2),  # Round to 2 decimal places
                    "last_date": last_date  # Add the last retrieved date
                }
        except Exception as e:
            invalid_tickers.append(ticker)

    response_data = {
        "is_valid": not bool(invalid_tickers),  # True if no invalid tickers
        "valid_tickers": valid_tickers,
    }

    if invalid_tickers:
        response_data["invalid_tickers"] = invalid_tickers

    return Response(response_data)

# For portfolio suggestion
@api_view(['POST'])
def optimize_portfolio_api(request):
    try:
        # Extract user inputs
        data = request.data  
        universe_tickers = data.get("tickers", ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'GOOG', 'ZOMATO.NS'])
        investment_budget_usd = data.get("budget", 10000)

        # Fetch USD to INR exchange rate
        exchange_rate_data = yf.download('USDINR=X', period='1d')
        exchange_rate = exchange_rate_data['Close'].iloc[-1] if not exchange_rate_data.empty else 83.0

        # Categorize stocks
        indian_stocks = [ticker for ticker in universe_tickers if ticker.endswith('.NS')]
        us_stocks = [ticker for ticker in universe_tickers if ticker not in indian_stocks]

        # Download historical data
        stock_data = yf.download(universe_tickers, period="5y")['Close']
        returns_all = stock_data.pct_change(fill_method=None).dropna()  # Fix for FutureWarning

        # Get latest stock prices
        latest_prices = yf.download(universe_tickers, period='1d', interval='1d')['Close'].iloc[-1].astype(float)
        latest_prices_usd = latest_prices.copy()

        # Convert INR prices to USD
        for ticker in indian_stocks:
            latest_prices_usd[ticker] /= float(exchange_rate.iloc[0])  # Fix for FutureWarning

        # Expected returns & covariance matrix
        exp_returns = returns_all.mean().astype(float)
        cov_matrix = returns_all.cov().astype(float)

        def optimize_portfolio(strategy):
            n = len(universe_tickers)
            if n == 0:
                return None

            w = cp.Variable(n)
            port_variance = cp.quad_form(w, cov_matrix.values)
            port_return = exp_returns.values @ w

            # Constraints
            constraints = [cp.sum(w) == 1, w >= 0.0001]

            if strategy == "max_sharpe":
                # DCP-compliant formulation for Sharpe ratio maximization
                risk_free_rate = 0.0  # Assuming risk-free rate is 0 for simplicity
                # Maximize the risk-adjusted return (Sharpe ratio)
                objective = cp.Maximize(port_return - 0.5 * port_variance)  # Quadratic approximation
            elif strategy == "min_volatility":
                objective = cp.Minimize(port_variance)
                risk_free_rate = 0.0  # Define risk_free_rate for other strategies
            else:
                objective = cp.Minimize(port_variance - 0.5 * port_return)
                constraints.append(w <= 0.4)
                risk_free_rate = 0.0  # Define risk_free_rate for other strategies

            problem = cp.Problem(objective, constraints)
            try:
                problem.solve(solver=cp.ECOS)  # Use ECOS solver
                if w.value is not None:
                    opt_weights = np.maximum(w.value, 0)
                    opt_weights /= np.sum(opt_weights)
                    return {
                        'tickers': universe_tickers,
                        'weights': opt_weights.tolist(),
                        'expected_return': float(port_return.value),
                        'risk': float(np.sqrt(port_variance.value)),
                        'sharpe_ratio': float((port_return.value - risk_free_rate) / np.sqrt(port_variance.value)) if np.sqrt(port_variance.value) > 0 else 0,
                        'risk_free_rate': risk_free_rate  # Pass risk_free_rate to the response
                    }
            except cp.error.SolverError as e:
                print(f"Solver Error for {strategy}: {e}")
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
                    "price": round(float(price_usd), 2),
                    "allocated": round(float(allocated), 2),
                    "spent": round(float(spent), 2),
                })

            return {
                "name": title,
                "stocks": portfolio['tickers'],
                "optimized_weights": [round(w, 2) for w in portfolio['weights']],  # Round to 2 decimal places
                "expected_daily_return": round(float(portfolio['expected_return']), 4),
                "portfolio_risk": round(float(portfolio['risk']), 4),
                "sharpe_ratio": round(float(portfolio['sharpe_ratio']), 4),
                "share_allocation": shares_allocated,
                "total_spent": round(total_spent, 2),
                "budget": investment_budget_usd
            }

        # Compute portfolios for different strategies
        strategies = ["max_sharpe", "min_volatility", "balanced"]
        portfolios = [create_portfolio_response(strategy.replace('_', ' ').title() + " Portfolio", optimize_portfolio(strategy)) for strategy in strategies]

        return Response([p for p in portfolios if p is not None])

    except Exception as e:
        print(f"Error: {e}")
        return Response({"error": str(e)}, status=500)
        


# from rest_framework.decorators import api_view
# from rest_framework.response import Response
# import yfinance as yf
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.linear_model import Ridge
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from datetime import datetime
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping

# Fetch stock data
def get_stock_data(ticker, start, end):
    stock = yf.download(ticker, start=start, end=end)
    return stock[['Close']]

# Prepare Data
def prepare_data(df, lookback=10):
    data = df.copy()
    for i in range(1, lookback + 1):
        data[f'Close_Lag_{i}'] = data['Close'].shift(i)
    data.dropna(inplace=True)
    return data

# Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Evaluate Model
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f'\n{model_name} Performance:')
    print(f'MAE: {mae:.4f}, MSE: {mse:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}')
    return mae, mse, mape, r2

# LSTM Preparation
def prepare_lstm_data(df, time_steps=60):
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

# Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

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

# Ridge Regression API
@api_view(['POST'])
def ridge_stock_prediction(request):
    try:
        data = request.data
        ticker = data.get("ticker", "AAPL")
        future_days = int(data.get("days", 30))

        # Fetch stock data
        end_date = datetime.today().strftime('%Y-%m-%d')
        df = get_stock_data(ticker, '2015-01-01', end_date)
        data = prepare_data(df)

        # Split Data
        X = data.drop(columns=['Close'])
        y = data['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Ridge Regression Model
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)

        # Predict on test data
        y_pred = model.predict(X_test_scaled)

        # Compute metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Prepare actual vs predicted data
        actual_vs_predicted = [
            {"date": data.index[len(X_train) + i].strftime("%Y-%m-%d"), "Actual": round(float(y_test.iloc[i]), 2),
             "Predicted": round(float(y_pred[i]), 2)}
            for i in range(len(y_test))
        ]

        # Predict future prices
        last_known_data = X_test.iloc[-1:].values
        future_predictions = predict_future_prices_ml(model, scaler, last_known_data, future_days)

        return Response({
            "ticker": ticker,
            "future_predictions": future_predictions,
            "metrics": {"MAE": round(mae, 4), "MSE": round(mse, 4), "MAPE": round(mape, 4), "R²": round(r2, 4)},
            "actual_vs_predicted": actual_vs_predicted
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)

# Helper function to predict future prices for Ridge, Random Forest, and XGBoost
def predict_future_prices_ml(model, scaler, last_known_data, prediction_days):
    future_predictions = []
    current_data = last_known_data.copy()
    
    for _ in range(prediction_days):
        if scaler:
            current_data_scaled = scaler.transform(current_data)
            next_day_pred = model.predict(current_data_scaled)
        else:
            next_day_pred = model.predict(current_data)
        
        future_predictions.append(next_day_pred[0])
        current_data = np.roll(current_data, -1)
        current_data[-1] = next_day_pred
    
    return future_predictions

# Random Forest API
@api_view(['POST'])
def random_forest_stock_prediction(request):
    try:
        data = request.data
        ticker = data.get("ticker", "AAPL")
        future_days = int(data.get("days", 30))

        # Fetch stock data
        end_date = datetime.today().strftime('%Y-%m-%d')
        df = get_stock_data(ticker, '2015-01-01', end_date)
        data = prepare_data(df)

        # Split Data
        X = data.drop(columns=['Close'])
        y = data['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # Train Random Forest Model
        model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
        model.fit(X_train, y_train)

        # Predict on test data
        y_pred = model.predict(X_test)

        # Compute metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Prepare actual vs predicted data
        actual_vs_predicted = [
            {"date": data.index[len(X_train) + i].strftime("%Y-%m-%d"), "Actual": round(float(y_test.iloc[i]), 2),
             "Predicted": round(float(y_pred[i]), 2)}
            for i in range(len(y_test))
        ]

        # Predict future prices
        last_known_data = X_test.iloc[-1:].values
        future_predictions = predict_future_prices_ml(model, None, last_known_data, future_days)

        return Response({
            "ticker": ticker,
            "future_predictions": future_predictions,
            "metrics": {"MAE": round(mae, 4), "MSE": round(mse, 4), "MAPE": round(mape, 4), "R²": round(r2, 4)},
            "actual_vs_predicted": actual_vs_predicted
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)

# XGBoost API
@api_view(['POST'])
def xgboost_stock_prediction(request):
    try:
        data = request.data
        ticker = data.get("ticker", "AAPL")
        future_days = int(data.get("days", 30))

        # Fetch stock data
        end_date = datetime.today().strftime('%Y-%m-%d')
        df = get_stock_data(ticker, '2015-01-01', end_date)
        data = prepare_data(df)

        # Split Data
        X = data.drop(columns=['Close'])
        y = data['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # Train XGBoost Model
        model = XGBRegressor(objective='reg:squarederror', n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)

        # Predict on test data
        y_pred = model.predict(X_test)

        # Compute metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Prepare actual vs predicted data
        actual_vs_predicted = [
            {"date": data.index[len(X_train) + i].strftime("%Y-%m-%d"), "Actual": round(float(y_test.iloc[i]), 2),
             "Predicted": round(float(y_pred[i]), 2)}
            for i in range(len(y_test))
        ]

        # Predict future prices
        last_known_data = X_test.iloc[-1:].values
        future_predictions = predict_future_prices_ml(model, None, last_known_data, future_days)

        return Response({
            "ticker": ticker,
            "future_predictions": future_predictions,
            "metrics": {"MAE": round(mae, 4), "MSE": round(mse, 4), "MAPE": round(mape, 4), "R²": round(r2, 4)},
            "actual_vs_predicted": actual_vs_predicted
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)

# LSTM API
@api_view(['POST'])
def lstm_stock_prediction(request):
    try:
        data = request.data
        ticker = data.get("ticker", "AAPL")
        future_days = int(data.get("days", 30))

        # Fetch stock data
        end_date = datetime.today().strftime('%Y-%m-%d')
        df = get_stock_data(ticker, '2015-01-01', end_date)

        # Prepare LSTM data
        X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(df)

        # Build and train LSTM model
        model = build_lstm_model((X_train.shape[1], 1))
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