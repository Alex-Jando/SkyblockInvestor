"use client";

import { useEffect, useMemo, useState } from "react";

type BasketItem = {
  item_id: string;
  weight_pct: number;
  expected_return: number;
  confidence: number;
  liquidity_score: number;
  spread_pct: number;
  max_alloc_pct_feasible: number;
  current_buy_price: number | null;
  current_sell_price: number | null;
};

type BasketResponse = {
  day: string | null;
  ts: string | null;
  decision_horizon_days: number;
  model_version: string | null;
  notes: string | null;
  items: BasketItem[];
};

const DEFAULT_COINS = 50_000_000;

function pct(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function num(value: number): string {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value);
}

export default function HomePage() {
  const [coins, setCoins] = useState<number>(DEFAULT_COINS);
  const [data, setData] = useState<BasketResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        const response = await fetch("/api/basket/latest", { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`Request failed: ${response.status}`);
        }
        const payload = (await response.json()) as BasketResponse;
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

  const rows = useMemo(() => data?.items ?? [], [data?.items]);

  return (
    <>
      <section className="panel">
        <h1 style={{ margin: "0 0 8px 0" }}>Latest BUY Basket</h1>
        <p className="muted" style={{ margin: "0 0 10px 0" }}>
          Decision horizon: {data?.decision_horizon_days ?? 7}d. Last updated:{" "}
          {data?.ts ? new Date(data.ts).toUTCString() : "N/A"}.
        </p>
        <p className="muted" style={{ marginTop: 0 }}>
          Model version: {data?.model_version ?? "N/A"}
        </p>
        <label htmlFor="coinsInput">Coins to invest</label>
        <div style={{ marginTop: 6 }}>
          <input
            id="coinsInput"
            type="number"
            min={0}
            step={1000000}
            value={coins}
            onChange={(event) => setCoins(Number(event.target.value))}
          />
        </div>
      </section>

      <section className="panel">
        <h2 style={{ marginTop: 0 }}>Basket Items</h2>
        {loading && <p className="muted">Loading basket...</p>}
        {error && <p className="danger">Error: {error}</p>}
        {!loading && !error && rows.length === 0 && (
          <p className="warn">No BUY signals are currently available.</p>
        )}
        {rows.length > 0 && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Item</th>
                  <th>Weight</th>
                  <th>Expected 7d</th>
                  <th>Confidence</th>
                  <th>Liquidity</th>
                  <th>Spread</th>
                  <th>Feasible Cap</th>
                  <th>Bid / Insta-sell</th>
                  <th>Ask / Insta-buy</th>
                  <th>Allocation</th>
                  <th>Qty</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((item) => {
                  const weight = Number(item.weight_pct);
                  const askPrice = Number(item.current_sell_price ?? 0);
                  const allocation = coins * weight;
                  const qty = askPrice > 0 ? allocation / askPrice : 0;
                  return (
                    <tr key={item.item_id}>
                      <td>{item.item_id}</td>
                      <td>{pct(weight)}</td>
                      <td>{pct(Number(item.expected_return))}</td>
                      <td>{pct(Number(item.confidence))}</td>
                      <td>{Number(item.liquidity_score).toFixed(2)}</td>
                      <td>{pct(Number(item.spread_pct))}</td>
                      <td>{pct(Number(item.max_alloc_pct_feasible))}</td>
                      <td>{num(Number(item.current_buy_price ?? 0))}</td>
                      <td>{num(Number(item.current_sell_price ?? 0))}</td>
                      <td>{num(allocation)}</td>
                      <td>{num(qty)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="panel">
        <h2 style={{ marginTop: 0 }}>Risk Notes</h2>
        <p className="muted" style={{ margin: 0 }}>
          {data?.notes ?? "No notes available."}
        </p>
      </section>
    </>
  );
}
