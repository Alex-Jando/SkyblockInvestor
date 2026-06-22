from __future__ import annotations

EPS = 1e-9

# Hypixel Bazaar quick_status naming is from the market's perspective:
# - buyPrice: highest buy order / bid. You receive this when you instant-sell.
# - sellPrice: lowest sell offer / ask. You pay this when you instant-buy.
# A realistic paper model must buy at ask and liquidate/mark at bid, not the
# other way around. The Bazaar tax is charged on sell-side proceeds.
BAZAAR_TAX = 0.0125


def bid_price(buy_price: float | int | None) -> float:
    return float(buy_price or 0.0)


def ask_price(sell_price: float | int | None) -> float:
    return float(sell_price or 0.0)


def mid_price(buy_price: float | int | None, sell_price: float | int | None) -> float:
    bid = bid_price(buy_price)
    ask = ask_price(sell_price)
    if bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return max(bid, ask)


def spread_pct(buy_price: float | int | None, sell_price: float | int | None) -> float:
    bid = bid_price(buy_price)
    ask = ask_price(sell_price)
    mid = mid_price(bid, ask)
    if bid <= 0 or ask <= 0 or mid <= EPS:
        return 0.0
    return max(0.0, (ask - bid) / mid)


def liquidation_value(qty: float, buy_price: float | int | None, tax: float = BAZAAR_TAX) -> float:
    """Coins received if qty is sold immediately into current buy orders."""
    return max(0.0, float(qty)) * bid_price(buy_price) * (1.0 - tax)


def entry_cost(qty: float, sell_price: float | int | None) -> float:
    """Coins paid if qty is bought immediately from current sell offers."""
    return max(0.0, float(qty)) * ask_price(sell_price)


def net_exit_return(entry_ask: float | int | None, exit_bid: float | int | None, tax: float = BAZAAR_TAX) -> float:
    ask = ask_price(entry_ask)
    bid = bid_price(exit_bid)
    if ask <= EPS or bid <= 0:
        return 0.0
    return (bid * (1.0 - tax) / ask) - 1.0
