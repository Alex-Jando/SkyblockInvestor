"use client";

import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

type PerfPoint = {
  day: string;
  cumulative_return_pct: number;
  daily_return_pct: number;
  max_drawdown_pct: number;
};

type PerfResponse = {
  series: PerfPoint[];
};

function pct(value: number): string {
  return `${value.toFixed(2)}%`;
}

export default function PerformancePage() {
  const [series, setSeries] = useState<PerfPoint[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        const response = await fetch("/api/performance", { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`Request failed: ${response.status}`);
        }
        const payload = (await response.json()) as PerfResponse;
        if (!cancelled) {
          setSeries(payload.series ?? []);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError((err as Error).message);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const stats = useMemo(() => {
    if (series.length === 0) {
      return {
        totalReturn: 0,
        maxDrawdown: 0,
        bestDay: 0,
        worstDay: 0
      };
    }
    const totalReturn = Number(series[series.length - 1].cumulative_return_pct);
    const maxDrawdown = Math.min(...series.map((point) => Number(point.max_drawdown_pct)));
    const bestDay = Math.max(...series.map((point) => Number(point.daily_return_pct)));
    const worstDay = Math.min(...series.map((point) => Number(point.daily_return_pct)));
    return { totalReturn, maxDrawdown, bestDay, worstDay };
  }, [series]);

  return (
    <>
      <section className="panel">
        <h1 style={{ margin: "0 0 8px 0" }}>Paper Portfolio Performance</h1>
        <p className="muted" style={{ margin: 0 }}>
          Cumulative return, daily return, and drawdown from the global daily rebalance portfolio.
        </p>
      </section>

      <section className="panel">
        {loading && <p className="muted">Loading performance...</p>}
        {error && <p className="danger">Error: {error}</p>}
        {!loading && !error && series.length === 0 && (
          <p className="muted">No performance data available yet.</p>
        )}
        {series.length > 0 && (
          <div style={{ width: "100%", height: 360 }}>
            <ResponsiveContainer>
              <LineChart data={series}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(31,43,42,0.2)" />
                <XAxis dataKey="day" tick={{ fill: "#546766", fontSize: 12 }} />
                <YAxis
                  tick={{ fill: "#546766", fontSize: 12 }}
                  tickFormatter={(value) => `${Number(value).toFixed(1)}%`}
                />
                <Tooltip
                  formatter={(value: number) => pct(Number(value))}
                  labelFormatter={(label) => `Day: ${label}`}
                />
                <Line
                  type="monotone"
                  dataKey="cumulative_return_pct"
                  stroke="#0f766e"
                  strokeWidth={2.4}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </section>

      <section className="panel">
        <div className="stats-grid">
          <div className="stat">
            <div className="stat-label">Total Return</div>
            <div className="stat-value">{pct(stats.totalReturn)}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Max Drawdown</div>
            <div className="stat-value">{pct(stats.maxDrawdown)}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Best Day</div>
            <div className="stat-value">{pct(stats.bestDay)}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Worst Day</div>
            <div className="stat-value">{pct(stats.worstDay)}</div>
          </div>
          <div className="stat">
            <div className="stat-label">Last Updated</div>
            <div className="stat-value">
              {series.length > 0 ? String(series[series.length - 1].day) : "N/A"}
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
