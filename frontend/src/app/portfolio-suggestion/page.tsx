"use client";

import { useState } from "react";
import Navbar from "@/components/Navbar";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";
import PortfolioCarosel from "@/components/PortfolioCarosel";
import Portfolio from "@/components/Portfolio";
import axios from "axios";

export default function Page() {
  const [stocks, setStocks] = useState("");
  const [budget, setBudget] = useState("");

  const resEx = [
    {
      name: "Max Sharpe Portfolio",
      stocks: ["AAPL", "MSFT", "NVDA"],
      optimized_weights: [0.0, 0.0, 0.0],
      expected_daily_return: 0,
      portfolio_risk: 0,
      sharpe_ratio: 0,
      share_allocation: [
        {
          symbol: "AAPL",
          shares: 0,
          price: 0,
          allocated: 0,
          spent: 0.0,
        },
        {
          symbol: "MSFT",
          shares: 0,
          price: 0,
          allocated: 0,
          spent: 0,
        },
        {
          symbol: "NVDA",
          shares: 0,
          price: 0,
          allocated: 0,
          spent: 0,
        },
      ],
      total_spent: 0,
      budget: 0,
    },
    {
      name: "Min Volatility Portfolio",
      stocks: ["AAPL", "MSFT", "NVDA"],
      optimized_weights: [0.0, 0.0, 0.0],
      expected_daily_return: 0,
      portfolio_risk: 0,
      sharpe_ratio: 0,
      share_allocation: [
        {
          symbol: "AAPL",
          shares: 0,
          price: 0,
          allocated: 0,
          spent: 0.0,
        },
        {
          symbol: "MSFT",
          shares: 0,
          price: 0,
          allocated: 0,
          spent: 0,
        },
        {
          symbol: "NVDA",
          shares: 0,
          price: 0,
          allocated: 0,
          spent: 0,
        },
      ],
      total_spent: 0,
      budget: 0,
    },
    {
      name: "Balanced Portfolio",
      stocks: ["AAPL", "MSFT", "NVDA"],
      optimized_weights: [0.0, 0.0, 0.0],
      expected_daily_return: 0,
      portfolio_risk: 0,
      sharpe_ratio: 0,
      share_allocation: [
        {
          symbol: "AAPL",
          shares: 0,
          price: 0,
          allocated: 0,
          spent: 0.0,
        },
        {
          symbol: "MSFT",
          shares: 0,
          price: 0,
          allocated: 0,
          spent: 0,
        },
        {
          symbol: "NVDA",
          shares: 0,
          price: 0,
          allocated: 0,
          spent: 0,
        },
      ],
      total_spent: 0,
      budget: 0,
    },
  ];

  const [portfolios, setPortfolios] = useState(resEx);

  const COLORS = [
    "#0088FE",
    "#00C49F",
    "#FFBB28",
    "#FF8042",
    "#A28AD4",
    "#FF6666",
    "#85C1E9",
  ];

  const balancedPortfolioData = portfolios[2].share_allocation.map(
    (allocation, index) => ({
      ...allocation,
      optimized_weight: parseFloat(
        (portfolios[2].optimized_weights[index] * 100).toFixed(2)
      ),
    })
  );

  const handlePredict = async () => {
    if (!stocks) {
      return toast.error("Please enter stock symbols");
    }
    if (!budget) {
      return toast.error("Please enter budget");
    }
    if (Number(budget) <= 0) {
      return toast.error("Budget must be greater than 0");
    }

    const stockSymbols = stocks.split(" ");

    if (stockSymbols.length < 3) {
      return toast.error("Please enter at least 3 stock symbols");
    }

    try {
      const response = await axios.post(
        `http://localhost:8000/api/validate-tickers/`,
        {
          stocks: stocks,
        }
      );

      if (response.status !== 200 || !response.data.is_valid) {
        return toast.error(
          `Invalid stock symbols: ${response.data.invalid_tickers.join(", ")}`
        );
      }

      await axios
        .post(`http://localhost:8000/api/portfolio-suggestion/`, {
          tickers: stockSymbols,
          budget: Number(budget),
        })
        .then((response) => {
          setPortfolios(response.data);
          return;
        })
        .catch((error) => {
          return toast.error("Failed to get portfolio suggestion");
        });
    } catch (e: any) {
      toast.error("Failed to validate stock symbols");
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-5 w-full gap-4 m-6">
      <div className="md:col-span-5">
        <Navbar />
      </div>

      {/* Inputs Section */}
      <Card className="w-full min-h-56 h-full rounded px-4 py-2 md:col-span-2">
        <CardHeader>
          <CardTitle>Inputs</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid w-full max-w-sm items-center gap-1.5">
            <Label htmlFor="email">Stock Symbol</Label>
            <Input
              placeholder="AAPL MSFT TSLA"
              value={stocks}
              onChange={(e) => setStocks(e.target.value)}
            />
            <div>*Note: Enter each stock symbol with space in between</div>
          </div>
          <div className="grid w-full max-w-sm items-center gap-1.5">
            <Label>Investment Budget (USD)</Label>
            <Input
              placeholder="5000"
              type="number"
              value={budget}
              onChange={(e) => setBudget(e.target.value)}
            />
          </div>
          <Button onClick={handlePredict} className="w-full">
            Get Suggestion
          </Button>
        </CardContent>
      </Card>

      {/* Portfolios */}
      <PortfolioCarosel portfolios={portfolios} />

      {/* Balanced portfolio */}
      <div className="w-full min-h-56 h-full rounded md:col-span-3 ">
        <Portfolio item={portfolios[2]} />
      </div>

      {/* Pie Chart for Balanced Portfolio */}
      <Card className="w-full min-h-56 h-full rounded md:col-span-2 px-4 py-2">
        <CardHeader>
          <CardTitle>Balanced Portfolio Allocation</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={balancedPortfolioData}
                dataKey="optimized_weight"
                nameKey="symbol"
                cx="50%"
                cy="50%"
                outerRadius={80}
                fill="#8884d8"
                label
              >
                {balancedPortfolioData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
