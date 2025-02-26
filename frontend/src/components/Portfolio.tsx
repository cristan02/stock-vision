"use client";

import { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
} from "@/components/ui/carousel";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

export default function Portfolio({ item }: any) {
  return (
    <Card className="w-[99%] rounded h-full">
      <CardHeader>
        <CardTitle>{item.name}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid w-full max-w-sm items-center gap-1.5">
          <div className="flex gap-2">
            <p className="w-fit whitespace-nowrap font-medium">Risk :</p>
            <div className="w-fit text-primary flex flex-wrap gap-2">
              {item.portfolio_risk}
            </div>
          </div>

          <div className="flex gap-2">
            <p className="w-fit whitespace-nowrap font-medium">
              Sharpe Ratio :
            </p>
            <div className="w-fit text-primary flex flex-wrap gap-2">
              {item.sharpe_ratio}
            </div>
          </div>

          <div className="flex gap-2">
            <p className="w-fit whitespace-nowrap font-medium">Total Spent :</p>
            <div className="w-fit text-primary flex flex-wrap gap-2">
              {item.total_spent}
            </div>
          </div>

          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Stock</TableHead>
                <TableHead>Price</TableHead>
                <TableHead>Shares</TableHead>
                <TableHead>Spent</TableHead>
              </TableRow>
            </TableHeader>

            <TableBody>
              {item.share_allocation.map((algo: any, index: any) => (
                <TableRow key={index}>
                  <TableCell>{algo.symbol}</TableCell>
                  <TableCell>{algo.price}</TableCell>
                  <TableCell>{algo.shares}</TableCell>
                  <TableCell>{algo.spent}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}
