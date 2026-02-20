import { NextResponse } from "next/server";

import { getPool } from "@/lib/db";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const pool = getPool();
    const result = await pool.query(
      `
      select day, cumulative_return_pct, daily_return_pct, max_drawdown_pct
      from paper_portfolio_equity
      order by day asc
      `
    );

    return NextResponse.json(
      {
        series: result.rows
      },
      { status: 200 }
    );
  } catch (error) {
    console.error("GET /api/performance failed", error);
    return NextResponse.json(
      { error: "Failed to load performance data." },
      { status: 500 }
    );
  }
}
