"use client";

"use client";

import { useState, useEffect } from "react";
import Navbar from "@/components/Navbar";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import axios from "axios";
import { toast } from "sonner";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { parse } from "node:path";

export default function Page() {
  const [stock, setStock] = useState("");
  const [days, setDays] = useState("");
  const [stockData, setStockData] = useState([{ date: "", price: 0 }]);
  const [algoMetrics, setAlgoMetrics] = useState([
    {
      name: "Random Forest",
      MAE: "-",
      MSE: "-",
      MAPE: "-",
      actual_vs_predicted: [],
      future_predictions: [],
    },
    {
      name: "SVM",
      MAE: "-",
      MSE: "-",
      MAPE: "-",
      actual_vs_predicted: [],
      future_predictions: [],
    },
    {
      name: "LSTM",
      MAE: "-",
      MSE: "-",
      MAPE: "-",
      actual_vs_predicted: [],
      future_predictions: [],
    },
  ]);
  const [bestAlgorithm, setBestAlgorithm] = useState<any>();

  const handleRandomForest = async () => {
    try {
      await axios
        .post("http://localhost:8000/api/predict-stock-rf/", {
          ticker: stock,
          days: parseInt(days),
        })
        .then((response) => {
          const { data } = response;
          setStockData(data.stockData);
          setAlgoMetrics((prev) => {
            const newMetrics = [...prev];
            newMetrics[0].MAE = data.metrics.MAE;
            newMetrics[0].MSE = data.metrics.MSE;
            newMetrics[0].MAPE = data.metrics.MAPE;
            newMetrics[0].actual_vs_predicted = data.actual_vs_predicted;
            newMetrics[0].future_predictions = data.future_predictions;
            return newMetrics;
          });

          toast.success("Random Forest Algorithm Completed");
        })
        .catch((error) => {
          toast.error("Error in Random Forest Algorithm");
        });
    } catch (e) {
      toast.error("Error in Random Forest Algorithm");
    }
  };

  const handleSVM = async () => {
    try {
      await axios
        .post("http://localhost:8000/api/predict-stock-svm/", {
          ticker: stock,
          days: parseInt(days),
        })
        .then((response) => {
          const { data } = response;
          console.log(data);
          setAlgoMetrics((prev) => {
            const newMetrics = [...prev];
            newMetrics[1].MAE = data.metrics.MAE;
            newMetrics[1].MSE = data.metrics.MSE;
            newMetrics[1].MAPE = data.metrics.MAPE;
            newMetrics[1].actual_vs_predicted = data.actual_vs_predicted;
            newMetrics[1].future_predictions = data.future_predictions;
            return newMetrics;
          });

          toast.success("SVM Algorithm Completed");
        })
        .catch((error) => {
          toast.error("Error in SVM Algorithm");
        });
    } catch (e) {
      toast.error("Error in SVM Algorithm");
    }
  };

  const handleLSTM = async () => {
    try {
      await axios
        .post("http://localhost:8000/api/predict-stock-lstm/", {
          ticker: stock,
          days: parseInt(days),
        })
        .then((response) => {
          const { data } = response;
          setAlgoMetrics((prev) => {
            const newMetrics = [...prev];
            newMetrics[2].MAE = data.metrics.MAE;
            newMetrics[2].MSE = data.metrics.MSE;
            newMetrics[2].MAPE = data.metrics.MAPE;
            newMetrics[2].actual_vs_predicted = data.actual_vs_predicted;
            newMetrics[2].future_predictions = data.future_predictions;
            return newMetrics;
          });

          toast.success("LSTM Algorithm Completed");
        })
        .catch((error) => {
          toast.error("Error in LSTM Algorithm");
        });
    } catch (e) {
      toast.error("Error in LSTM Algorithm");
    }
  };

  const handlePredict = async () => {
    if (days === "") {
      toast.error("Please enter the number of days for prediction");
      return;
    }
    if (stock === "") {
      toast.error("Please enter the stock symbol");
      return;
    }
    if (parseInt(days) < 1) {
      toast.error("Please enter a valid number of days for prediction");
      return;
    }

    try {
      const response = await axios.post(
        `http://localhost:8000/api/validate-tickers/`,
        {
          stocks: stock,
        }
      );

      if (response.status !== 200 || !response.data.is_valid) {
        toast.error(
          `Invalid stock symbol: ${response.data.invalid_tickers.join(", ")}`
        );
        return;
      }
      handleRandomForest();
      handleSVM();
      handleLSTM();
    } catch (e) {
      toast.error("Failed to validate stock symbols");
      return;
    }
  };

  useEffect(() => {
    const validMetrics = algoMetrics.filter(
      (metric) => !isNaN(parseFloat(metric.MAE))
    );

    if (validMetrics.length > 0) {
      const bestAlgo = validMetrics.reduce((prev, current) =>
        parseFloat(prev.MAE) < parseFloat(current.MAE) ? prev : current
      );
      setBestAlgorithm(bestAlgo);
    }
  }, [algoMetrics]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-5 w-full gap-4 m-6">
      <div className="md:col-span-5">
        <Navbar />
      </div>

      {/* Inputs Section */}
      <Card className="w-full min-h-56 h-full rounded md:col-span-2 px-4 py-2">
        <CardHeader>
          <CardTitle>Inputs</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid w-full max-w-sm items-center gap-1.5">
            <Label htmlFor="email">Stock Symbol</Label>
            <Input
              placeholder="Enter Stock Symbol (e.g., AAPL)"
              value={stock}
              onChange={(e) => setStock(e.target.value.replace(/\s/g, ""))}
            />
          </div>
          <div className="grid w-full max-w-sm items-center gap-1.5">
            <Label htmlFor="email">No. of Days</Label>
            <Input
              placeholder="Days for Prediction (e.g., 7)"
              type="number"
              value={days}
              onChange={(e) => {
                const value = e.target.value;
                const intValue = parseInt(value, 10);

                if (!isNaN(intValue) && intValue > 0) {
                  setDays(intValue.toString());
                }
              }}
            />
          </div>
          <Button onClick={handlePredict} className="w-full">
            Predict Price
          </Button>
        </CardContent>
      </Card>

      {/* Algorithm Comparisons */}
      <Card className="w-full min-h-56 h-full rounded md:col-span-3 px-4 py-2">
        <CardHeader>
          <CardTitle>Algorithm Comparisons</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Algorithm</TableHead>
                <TableHead>MAE</TableHead>
                <TableHead>MSE</TableHead>
                <TableHead>MAPE</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {algoMetrics.map((algo, index) => (
                <TableRow key={index}>
                  <TableCell>{algo.name}</TableCell>
                  <TableCell>{algo.MAE}</TableCell>
                  <TableCell>{algo.MSE}</TableCell>
                  <TableCell>{algo.MAPE}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Stock Closing Price Graph */}
      <Card className="w-full min-h-56 h-full rounded md:col-span-3 px-4 py-2">
        <CardHeader>
          <CardTitle>
            Closing Price Graph for {bestAlgorithm ? bestAlgorithm.name : "-"}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart
              data={
                bestAlgorithm
                  ? bestAlgorithm.actual_vs_predicted.slice(-50)
                  : []
              }
              margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="Actual"
                stroke="#8884d8"
                strokeWidth={2}
              />
              <Line
                type="monotone"
                dataKey="Predicted"
                stroke="#82ca9d"
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Best Algorithm Output */}
      <Card className="w-full min-h-56 h-full rounded md:col-span-2 px-4 py-2">
        <CardHeader>
          <CardTitle>Results</CardTitle>
        </CardHeader>
        <CardContent>
          <div className=" grid gap-2">
            <div className=" flex gap-2 items-center">
              <p className=""> Best Algorithm : </p>
              <div className=" w-fit text-primary">
                {bestAlgorithm ? bestAlgorithm.name : "-"}
              </div>
            </div>

            <div className=" flex gap-2 items-center">
              <p className=" ">MAE :</p>
              <div className=" w-fit text-primary">
                {bestAlgorithm ? bestAlgorithm.MAE.toFixed(4) : "-"}
              </div>
            </div>

            <div className=" flex gap-2 items-center">
              <p className=" ">Predicted price :</p>
              <div className=" w-fit flex justify-center items-center">
                <div className=" bg-secondary p-2 rounded-full ">
                  {bestAlgorithm
                    ? bestAlgorithm.future_predictions.at(-1).toFixed(2)
                    : "-"}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
