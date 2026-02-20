export default function HowPage() {
  return (
    <>
      <section className="panel">
        <h1 style={{ margin: "0 0 8px 0" }}>How This MVP Works</h1>
        <p className="muted" style={{ margin: 0 }}>
          This app recommends a daily-rebalanced Bazaar basket for 24h to 30d horizons and shows paper
          performance in percent terms.
        </p>
      </section>

      <section className="panel">
        <h2 style={{ marginTop: 0 }}>Pipeline</h2>
        <ul style={{ margin: 0, paddingLeft: 18 }}>
          <li>Fetches all Bazaar products once per day from the Hypixel API.</li>
          <li>Stores one upserted snapshot per item per UTC day.</li>
          <li>Builds rolling features: returns, volatility, spread, liquidity, volume trend, imbalance.</li>
          <li>Predicts expected returns for 1d, 3d, 7d, 14d, and 30d horizons.</li>
          <li>Applies blacklist + risk filters + feasibility constraints before recommending BUY items.</li>
          <li>Builds SELL list for expected downside over medium horizons.</li>
          <li>Simulates a single global paper portfolio with full daily rebalance.</li>
        </ul>
      </section>

      <section className="panel">
        <h2 style={{ marginTop: 0 }}>Core Assumptions</h2>
        <ul style={{ margin: 0, paddingLeft: 18 }}>
          <li>Only Bazaar items are considered.</li>
          <li>Buying uses insta-buy (`buy_price`), selling uses insta-sell (`sell_price`).</li>
          <li>Portfolio starts at 100,000,000 coins and is evaluated as percent return over time.</li>
          <li>Each position has both a model cap and a liquidity/turnover feasibility cap.</li>
          <li>Any basket weight under 5% is dropped before final normalization.</li>
        </ul>
      </section>

      <section className="panel">
        <h2 style={{ marginTop: 0 }}>Not Included In MVP</h2>
        <ul style={{ margin: 0, paddingLeft: 18 }}>
          <li>No authentication or user-specific portfolios.</li>
          <li>No intraday execution logic (daily snapshots only).</li>
          <li>No claim of live tradeability guarantees, only feasibility proxies.</li>
        </ul>
      </section>
    </>
  );
}
