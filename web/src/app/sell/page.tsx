"use client";

import { useEffect, useState } from "react";

type SellItem = {
  item_id: string;
  expected_return: number;
  confidence: number;
  liquidity_score: number;
  spread_pct: number;
  current_buy_price: number | null;
  current_sell_price: number | null;
};

type SellResponse = {
  day: string | null;
  ts: string | null;
  model_version: string | null;
  notes: string | null;
  items: SellItem[];
};

function pct(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

export default function SellPage() {
  const [data, setData] = useState<SellResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        const response = await fetch("/api/sell/latest", { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`Request failed: ${response.status}`);
        }
        const payload = (await response.json()) as SellResponse;
        if (!cancelled) {
          setData(payload);
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

  return (
    <>
      <section className="panel">
        <h1 style={{ margin: "0 0 8px 0" }}>Latest SELL Signals</h1>
        <p className="muted" style={{ margin: "0 0 4px 0" }}>
          Last updated: {data?.ts ? new Date(data.ts).toUTCString() : "N/A"}
        </p>
        <p className="muted" style={{ marginTop: 0 }}>
          Model version: {data?.model_version ?? "N/A"}
        </p>
      </section>

      <section className="panel">
        {loading && <p className="muted">Loading sell list...</p>}
        {error && <p className="danger">Error: {error}</p>}
        {!loading && !error && (data?.items?.length ?? 0) === 0 && (
          <p className="muted">No SELL signals for the latest basket.</p>
        )}
        {(data?.items?.length ?? 0) > 0 && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Item</th>
                  <th>Expected 7d</th>
                  <th>Confidence</th>
                  <th>Liquidity</th>
                  <th>Spread</th>
                  <th>Buy Price</th>
                  <th>Sell Price</th>
                </tr>
              </thead>
              <tbody>
                {data?.items.map((item) => (
                  <tr key={item.item_id}>
                    <td>{item.item_id}</td>
                    <td className="danger">{pct(Number(item.expected_return))}</td>
                    <td>{pct(Number(item.confidence))}</td>
                    <td>{Number(item.liquidity_score).toFixed(2)}</td>
                    <td>{pct(Number(item.spread_pct))}</td>
                    <td>{Number(item.current_buy_price ?? 0).toFixed(2)}</td>
                    <td>{Number(item.current_sell_price ?? 0).toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="panel">
        <h2 style={{ marginTop: 0 }}>Notes</h2>
        <p className="muted" style={{ margin: 0 }}>
          {data?.notes ?? "No notes available."}
        </p>
      </section>
    </>
  );
}
