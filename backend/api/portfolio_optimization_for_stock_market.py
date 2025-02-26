from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import numpy as np
import pandas as pd
import math
import cvxpy as cp
import yfinance as yf

@csrf_exempt
def optimize_portfolio_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)
    
    try:
        data = json.loads(request.body)
        universe_tickers = data.get("universe_tickers", [])
        investment_budget_usd = data.get("investment_budget_usd", 5000)
        
        if not universe_tickers:
            return JsonResponse({"error": "No tickers provided"}, status=400)
        
        # Fetch exchange rate
        exchange_rate = yf.download('USDINR=X', period='1d')['Close'].iloc[-1] if 'USDINR=X' in yf.download('USDINR=X', period='1d') else 83.0

        # Categorize stocks
        indian_stocks = [ticker for ticker in universe_tickers if ticker.endswith('.NS')]
        us_stocks = [ticker for ticker in universe_tickers if ticker not in indian_stocks]

        # Download historical data
        data = yf.download(universe_tickers, period="5y")['Close']
        returns_all = data.pct_change().dropna()

        # Get latest stock prices
        latest_prices = yf.download(universe_tickers, period='1d', interval='1d')['Close'].iloc[-1]
        latest_prices_usd = latest_prices.copy()

        # Convert INR prices to USD
        for ticker in indian_stocks:
            latest_prices_usd[ticker] /= exchange_rate

        # Expected returns & covariance matrix
        exp_returns = returns_all.mean()
        cov_matrix = returns_all.cov()
    
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
            else:  # Balanced
                objective = cp.Minimize(port_variance - 0.5 * (exp_returns.values @ w))
                constraints.append(w <= 0.4)

            problem = cp.Problem(objective, constraints)
            try:
                problem.solve()
                if w.value is not None:
                    opt_weights = np.maximum(w.value, 0)
                    opt_weights /= np.sum(opt_weights)
                    return {
                        'tickers': universe_tickers,
                        'weights': opt_weights.tolist(),
                        'expected_return': float(exp_returns.values @ opt_weights),
                        'risk': float(np.sqrt(port_variance.value)),
                        'sharpe_ratio': float((exp_returns.values @ opt_weights) / np.sqrt(port_variance.value)) if np.sqrt(port_variance.value) > 0 else 0
                    }
            except cp.error.SolverError:
                return None

        def format_portfolio_response(name, portfolio):
            allocated_amounts = investment_budget_usd * np.array(portfolio['weights'])
            share_allocation, total_spent = [], 0

            for ticker, allocated in zip(portfolio['tickers'], allocated_amounts):
                price_usd = latest_prices_usd.get(ticker, np.nan)
                if pd.isna(price_usd):
                    continue
                num_shares = math.floor(allocated / price_usd)
                spent = num_shares * price_usd
                total_spent += spent
                share_allocation.append({
                    "symbol": ticker,
                    "shares": num_shares,
                    "price": round(price_usd, 2),
                    "allocated": round(allocated, 2),
                    "spent": round(spent, 2)
                })

            return {
                "name": name,
                "stocks": portfolio['tickers'],
                "optimized_weights": portfolio['weights'],
                "expected_daily_return": round(portfolio['expected_return'], 6),
                "portfolio_risk": round(portfolio['risk'], 6),
                "sharpe_ratio": round(portfolio['sharpe_ratio'], 2),
                "share_allocation": share_allocation,
                "total_spent": round(total_spent, 2),
                "budget": investment_budget_usd
            }
    
        portfolios = []
        for strategy in ["max_sharpe", "min_volatility", "balanced"]:
            portfolio = optimize_portfolio(strategy)
            if portfolio:
                portfolios.append(format_portfolio_response(strategy.replace('_', ' ').title() + " Portfolio", portfolio))
        
        return JsonResponse(portfolios, safe=False)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
