import { NextResponse } from "next/server";

import { getPool } from "@/lib/db";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const pool = getPool();
    const basketResult = await pool.query(
      `
      select id, day, ts, decision_horizon_days, model_version, notes
      from baskets
      order by day desc
      limit 1
      `
    );

    if (basketResult.rowCount === 0) {
      return NextResponse.json(
        {
          day: null,
          ts: null,
          decision_horizon_days: 7,
          model_version: null,
          notes: "No basket available yet.",
          items: []
        },
        { status: 200 }
      );
    }

    const basket = basketResult.rows[0];
    const itemsResult = await pool.query(
      `
      select
        bi.item_id,
        bi.weight_pct,
        bi.expected_return,
        bi.confidence,
        bi.liquidity_score,
        bi.spread_pct,
        bi.max_alloc_pct_feasible,
        bs.buy_price as current_buy_price,
        bs.sell_price as current_sell_price
      from basket_items bi
      left join bazaar_snapshots bs
        on bs.item_id = bi.item_id
       and bs.day = $2
      where bi.basket_id = $1
        and bi.action = 'BUY'
      order by bi.weight_pct desc, bi.expected_return desc
      `,
      [basket.id, basket.day]
    );

    return NextResponse.json(
      {
        day: basket.day,
        ts: basket.ts,
        decision_horizon_days: basket.decision_horizon_days,
        model_version: basket.model_version,
        notes: basket.notes,
        items: itemsResult.rows
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("GET /api/basket/latest failed", error);
    return NextResponse.json(
      { error: "Failed to load latest basket." },
      { status: 500 }
    );
  }
}
