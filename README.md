# Hypixel SkyBlock Bazaar Investment Basket (MVP)

MVP stack:
- Frontend: Next.js (TypeScript) in `web/`
- Database: Supabase Postgres with SQL migrations in `supabase/migrations/`
- Worker: Python 3.11+ in `worker/`
- Scheduler: GitHub Actions cron in `.github/workflows/cron.yml`

## Repository Structure

```
/web
  package.json
  next.config.js
  src/app/...
/worker
  requirements.txt
  main.py
  hypixel_api.py
  db.py
  features.py
  model.py
  allocator.py
  portfolio.py
  risk_blacklist.json
/supabase/migrations
  001_init.sql
/.github/workflows
  cron.yml
README.md
.env.example
```

## 1) Create Supabase Project

1. Create a Supabase project.
2. Copy the Postgres connection string (URI), preferably the Supabase transaction pooler URI.
3. Save it as:
   - `SUPABASE_DATABASE_URL` for the worker.
   - `DATABASE_URL` for the Next.js server (can be the same value).
4. Recommended format:

```text
postgresql://postgres.<project_ref>:<db_password>@aws-1-us-east-1.pooler.supabase.com:6543/postgres?sslmode=require
```

## 2) Apply SQL Migration

Run from repo root:

```bash
psql "$SUPABASE_DATABASE_URL" -f supabase/migrations/001_init.sql
```

## 3) Configure Environment Variables

Copy `.env.example` to `.env` and set values:

- `HYPIXEL_API_KEY`
- `SUPABASE_DATABASE_URL`
- `DATABASE_URL`
- `PAPER_START_COINS` (default `100000000`)
- `SPREAD_MAX` (default `0.05`)
- `LIQUIDITY_MIN` (default `2.0`)
- `VOL_MAX` (default `0.25`)
- `VOLUME_DROP_FRAC` (default `0.2`)
- `FEASIBILITY_FACTOR` (default `0.05`)
- `TURNOVER_MIN_FRAC` (default `0.05`)
- `TURNOVER_CAP_FACTOR` (default `0.25`)
- `LIQUIDITY_TARGET` (default `2.5`)
- `MIN_EXPECTED_RETURN_BUY` (default `0.01`)
- `CONF_MIN_BUY` (default `0.55`)
- `MIN_WEIGHT_PCT` (default `0.05`)
- `MAX_WEIGHT_PCT` (default `0.30`)
- `MIN_BASKET_SIZE` (default `6`)
- `SELL_NEG_THRESHOLD` (default `-0.01`)
- `SELL_NEG_THRESHOLD_14D` (default `-0.01`)

## 4) Run Worker Manually (seed data)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r worker/requirements.txt
python worker/main.py
```

The worker is idempotent by day:
- snapshots upsert on `(item_id, day)`
- latest day basket replaces basket items
- latest day equity/holdings are overwritten

Bootstrap behavior:
- If history is under 15 days, thresholds are auto-relaxed.
- If history is under 30 days, moderate relaxations remain active.

Quick smoke test:

```bash
python -m worker.smoke_test
```

## 5) Run Web Locally

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:3000`.

## 6) Deploy Web to Vercel

1. Import the repo in Vercel.
2. Set root directory to `web`.
3. Set environment variable:
   - `DATABASE_URL` (same pooler URI you use locally).

## 7) Configure GitHub Actions Daily Cron

Add repository secrets:
- `HYPIXEL_API_KEY`
- `SUPABASE_DATABASE_URL` (pooler URI)

Workflow: `.github/workflows/cron.yml`  
Schedule: daily at `08:00 UTC` (plus manual `workflow_dispatch`).

## API Endpoints (Web)

- `GET /api/basket/latest`
- `GET /api/sell/latest`
- `GET /api/performance`

The web app only reads precomputed DB rows; no heavy modeling runs in the web server.

## Price Semantics / Profit Realism

Hypixel Bazaar `quick_status` names are easy to misread:

- `buyPrice` is the current buy-order side / bid. This is what you receive when you instant-sell.
- `sellPrice` is the current sell-offer side / ask. This is what you pay when you instant-buy.

Paper performance must therefore buy at `sellPrice` and mark or exit at `buyPrice` after Bazaar tax.
Older experiments that effectively bought at `buyPrice` and sold at `sellPrice` were spread-inverted and
overstated profit. Treat current paper results as conservative execution estimates, not confirmed real-money P&L.
