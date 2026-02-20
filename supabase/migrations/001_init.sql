create table if not exists bazaar_snapshots (
  id bigserial primary key,
  ts timestamptz not null,
  day date not null,
  item_id text not null,
  buy_price double precision not null,
  sell_price double precision not null,
  buy_volume double precision null,
  sell_volume double precision null,
  mid_price double precision not null,
  unique (item_id, day)
);

create index if not exists idx_bazaar_snapshots_item_day on bazaar_snapshots (item_id, day);
create index if not exists idx_bazaar_snapshots_day_desc on bazaar_snapshots (day desc);

create table if not exists item_signals (
  id bigserial primary key,
  ts timestamptz not null,
  day date not null,
  horizon_days int not null,
  item_id text not null,
  expected_return double precision not null,
  confidence double precision not null,
  liquidity_score double precision not null,
  spread_pct double precision not null,
  imbalance double precision not null,
  volatility_30d double precision not null,
  max_alloc_pct_feasible double precision not null,
  model_version text not null
);

create index if not exists idx_item_signals_day_desc on item_signals (day desc);
create index if not exists idx_item_signals_item_day_desc on item_signals (item_id, day desc);
create index if not exists idx_item_signals_horizon_day_desc on item_signals (horizon_days, day desc);

create table if not exists baskets (
  id bigserial primary key,
  ts timestamptz not null,
  day date not null,
  decision_horizon_days int not null default 7,
  model_version text not null,
  notes text null,
  unique (day)
);

create index if not exists idx_baskets_day_desc on baskets (day desc);

create table if not exists basket_items (
  id bigserial primary key,
  basket_id bigint not null references baskets(id) on delete cascade,
  item_id text not null,
  action text not null check (action in ('BUY', 'SELL')),
  weight_pct double precision not null default 0,
  expected_return double precision not null,
  confidence double precision not null,
  liquidity_score double precision not null,
  spread_pct double precision not null,
  max_alloc_pct_feasible double precision not null
);

create index if not exists idx_basket_items_basket_id on basket_items (basket_id);
create index if not exists idx_basket_items_action on basket_items (action);
create index if not exists idx_basket_items_item_id on basket_items (item_id);

create table if not exists paper_portfolio_equity (
  id bigserial primary key,
  ts timestamptz not null,
  day date not null unique,
  equity_value double precision not null,
  cash_value double precision not null,
  holdings_value double precision not null,
  cumulative_return_pct double precision not null,
  daily_return_pct double precision not null,
  max_drawdown_pct double precision not null
);

create table if not exists paper_portfolio_holdings (
  id bigserial primary key,
  day date not null,
  item_id text not null,
  qty double precision not null,
  cost_basis double precision not null,
  market_value double precision not null,
  unique (day, item_id)
);

create index if not exists idx_paper_holdings_day_desc on paper_portfolio_holdings (day desc);
create index if not exists idx_paper_holdings_item_id on paper_portfolio_holdings (item_id);

create table if not exists app_state (
  key text primary key,
  value text not null
);
