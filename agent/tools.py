"""Homework 1: the remaining commerce-agent tools.

The three lecture tools (`search_help_center`, `get_order`, `issue_refund`)
are implemented in agent/agent.py and are worked examples of the pattern:
check permissions first, go through agent/db.py for data, and return a
structured dict, never a prose error. The homework tools follow the same
pattern. agent/agent.py already wraps each function below as an SDK tool, so
once a function works here it works in chat with no further wiring.

Result convention (see agent/auth.py):
  - Success: a dict with "ok": True plus the payload fields named in each
    docstring.
  - Failure: {"ok": False, "error": <code>, "reason": <human-readable str>}.

Run the contract tests with: uv run pytest tests/test_hw_holes.py -k hw1
They are marked xfail and flip to passing as you implement each function.
"""

from __future__ import annotations

from typing import Any

from agent import db
from agent.auth import AuthContext, can_cancel_order, can_view_order, permission_denied
from agent.config import load_facts
from agent.helpcenter import load_policy_docs
from agent.killswitch import kill_switch
from seed.eligibility import effective_return_window_days

MAX_SEARCH_LIMIT = 25
DEFAULT_ORDER_LIMIT = 20


def get_policy(ctx: AuthContext, policy_id: str) -> dict[str, Any]:
    """Fetch one policy doc by its exact id. Risk tier: read.

    Every role may read every policy doc (the corpus is public help-center
    content), so this tool needs no permission check.

    Args:
        ctx: The caller's auth context. Unused here, but every tool takes it.
        policy_id: An exact policy id, e.g. "cw-returns" or
            "store-juniper-home-goods-policy". Matching is exact and
            case-sensitive; ids are the `policy_id` front-matter field of the
            files in data/policies/.

    Returns:
        On success: {"ok": True, "policy_id": str, "title": str,
        "audience": str, "body": str} where body is the markdown body of the
        doc without the front matter.
        If no doc has that id: {"ok": False, "error": "not_found",
        "reason": ...} naming the id that was requested.

    Implementation notes:
        agent.helpcenter.load_policy_docs() returns every parsed doc.
    """
    for doc in load_policy_docs():
        if doc.policy_id == policy_id:
            return {
                "ok": True,
                "policy_id": doc.policy_id,
                "title": doc.title,
                "audience": doc.audience,
                "body": doc.body,
            }
    return {
        "ok": False,
        "error": "not_found",
        "reason": f"no policy doc with id '{policy_id}'",
    }


def search_products(
    ctx: AuthContext,
    query: str,
    store: str | None = None,
    max_price_usd: float | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Search the product catalog. Risk tier: read.

    Every role may search products. Matching is deterministic keyword
    matching, not semantic search: a product matches when every whitespace
    token of `query` appears case-insensitively as a substring of the
    product's title or description.

    Args:
        ctx: The caller's auth context.
        query: Free-text query. Must be non-empty after stripping whitespace;
            otherwise return {"ok": False, "error": "invalid_argument",
            "reason": ...}.
        store: Optional store filter. Matched with
            agent.db.get_store_by_name (case-insensitive name or slug). If
            given and no store matches, return {"ok": False, "error":
            "not_found", "reason": ...} naming the store string.
        max_price_usd: Optional inclusive price ceiling. If given and not
            strictly positive, return an "invalid_argument" error.
        limit: Maximum products to return. Clamp to the range
            [1, MAX_SEARCH_LIMIT]; do not error on out-of-range values.

    Returns:
        {"ok": True, "products": [...], "count": <len(products)>} where each
        product is {"product_id": int, "store_id": int, "title": str,
        "price_usd": float}. Sort matches by price_usd ascending, then by
        product_id ascending, and truncate to `limit`. No matches is still a
        success: {"ok": True, "products": [], "count": 0}.

    Implementation notes:
        agent.db.list_products(conn, store_id) gives the candidate set.
        Use `with db.connection() as conn:` to close the database automatically.
    """
    query = query.strip()
    if not query:
        return {"ok": False, "error": "invalid_argument", "reason": "query must not be empty"}
    if max_price_usd is not None and max_price_usd <= 0:
        return {
            "ok": False,
            "error": "invalid_argument",
            "reason": f"max_price_usd must be positive, got {max_price_usd}",
        }
    limit = max(1, min(limit, MAX_SEARCH_LIMIT))

    with db.connection() as conn:
        store_id = None
        if store is not None:
            found_store = db.get_store_by_name(conn, store)
            if found_store is None:
                return {"ok": False, "error": "not_found", "reason": f"no store named '{store}'"}
            store_id = found_store.id
        candidates = db.list_products(conn, store_id)

    tokens = query.lower().split()
    matches = [
        p
        for p in candidates
        if all(t in f"{p.title} {p.description}".lower() for t in tokens)
        and (max_price_usd is None or p.price_usd <= max_price_usd)
    ]
    matches.sort(key=lambda p: (p.price_usd, p.id))
    matches = matches[:limit]
    products = [
        {
            "product_id": p.id,
            "store_id": p.store_id,
            "title": p.title,
            "price_usd": p.price_usd,
        }
        for p in matches
    ]
    return {"ok": True, "products": products, "count": len(products)}


def list_my_orders(ctx: AuthContext) -> dict[str, Any]:
    """List recent orders in the caller's own scope. Risk tier: read.

    Role behavior, straight from the access matrix in SPEC.md:
        - shopper: the caller's own orders.
        - merchant: the caller's store's orders (ctx.store_id).
        - support: support staff have no orders of their own and look up
          specific orders with get_order instead, so return {"ok": False,
          "error": "invalid_argument", "reason": ...} saying exactly that.

    Returns:
        For shopper and merchant: {"ok": True, "orders": [...],
        "count": <len(orders)>} where each order is
        agent.db.Order.to_public_dict() and the list holds at most
        DEFAULT_ORDER_LIMIT orders, newest first (agent.db.list_orders_for_user
        and list_orders_for_store already sort and limit this way).

    Implementation notes:
        No permission check is needed beyond the role dispatch, because the
        scope is baked into which query you run. That is the point of the
        tool: the model cannot ask for someone else's orders through it.
    """
    if ctx.role == "support":
        return {
            "ok": False,
            "error": "invalid_argument",
            "reason": "support staff have no orders of their own; use get_order for a specific order",
        }
    with db.connection() as conn:
        if ctx.role == "shopper":
            orders = db.list_orders_for_user(conn, ctx.user_id, limit=DEFAULT_ORDER_LIMIT)
        else:
            orders = db.list_orders_for_store(conn, ctx.store_id, limit=DEFAULT_ORDER_LIMIT)
    payload = [o.to_public_dict() for o in orders]
    return {"ok": True, "orders": payload, "count": len(payload)}


def cancel_order(ctx: AuthContext, order_id: int, reason: str) -> dict[str, Any]:
    """Cancel an order. Risk tier: write.

    This is the homework's write tool, and it must enforce two independent
    rules in this order:

    1. The access matrix (scope): use agent.auth.can_cancel_order. Shoppers
       may cancel only their own orders, merchants only their own store's
       orders, support any order. On failure return
       agent.auth.permission_denied(...) with a reason naming the role and
       the order id. Scope is checked before the status rule so that an
       out-of-scope caller learns nothing about the order's state.
    2. The pre-shipment rule (facts.yaml `cancel_cutoff`): only orders whose
       status is exactly "placed" can be cancelled, for every role. If the
       order is in scope but its status is not "placed", return
       {"ok": False, "error": "not_eligible", "reason": ...} that names the
       current status and states that orders can be cancelled only before
       shipment.

    Args:
        ctx: The caller's auth context.
        order_id: The order to cancel.
        reason: Free-text reason from the user; not validated.

    Returns:
        If no order has this id: {"ok": False, "error": "not_found",
        "reason": ...}.
        On success: {"ok": True, "order_id": order_id, "status": "cancelled"}
        after persisting the new status with agent.db.set_order_status.

    Implementation notes:
        Fetch with agent.db.get_order. Note the argument order of
        can_cancel_order(ctx, order_user_id, order_store_id).

    The Module 4 kill switch is checked first (before the scope and
    status rules and before your code), so that a paused write tool touches
    nothing. It is provided; the default ("off") returns None and falls
    through to your implementation.
    """
    paused = kill_switch("cancel_order")
    if paused is not None:
        return {"ok": False, "error": "paused", "reason": paused}
    with db.connection() as conn:
        order = db.get_order(conn, order_id)
        if order is None:
            return {"ok": False, "error": "not_found", "reason": f"no order #{order_id}"}
        if not can_cancel_order(ctx, order.user_id, order.store_id):
            return permission_denied(
                f"role '{ctx.role}' (user {ctx.user_id}) may not cancel order #{order_id}"
            )
        if order.status != "placed":
            return {
                "ok": False,
                "error": "not_eligible",
                "reason": (
                    f"order #{order_id} has status '{order.status}'; "
                    f"orders can only be cancelled before shipment"
                ),
            }
        db.set_order_status(conn, order_id, "cancelled")
        return {"ok": True, "order_id": order_id, "status": "cancelled"}


def find_order(ctx: AuthContext, query: str) -> dict[str, Any]:
    """Search the caller's orders by product name. Risk tier: read.

    Takes a natural-language query (e.g., "earmuffs I bought last week")
    and searches the authenticated user's orders for products whose name
    matches. Use fuzzy string matching (e.g., thefuzz.fuzz.partial_ratio
    or case-insensitive substring matching) to find orders whose product name is close to the
    query.

    Access rules: a shopper searches only the shopper's own orders, a
    merchant searches orders from the merchant's store, and support staff
    can search any orders. Use agent.db.list_order_search_candidates with
    user_id=ctx.user_id for shoppers, store_id=ctx.store_id for merchants,
    or all_orders=True only for support. Derive the scope from ctx, never
    from the query; reject unsupported roles or missing required identity.
    Use agent.db.list_products to map product IDs to product titles.

    The helper returns the complete authorised scope, newest first with
    order ID descending as the tie-breaker. Match product names first,
    preserve that order, then return at most five matches. Do not search
    only the 20 most recent orders. Convert matches with to_public_dict().

    Args:
        ctx: The caller's auth context.
        query: A natural-language description of the product.

    Returns:
        {"ok": True, "orders": [...]} with a list of matching orders
        (at most 5), each as the dict returned by agent.db. If no orders
        match, return {"ok": True, "orders": []}.
    """
    if ctx.role == "shopper":
        scope_kwargs: dict[str, Any] = {"user_id": ctx.user_id}
    elif ctx.role == "merchant":
        if ctx.store_id is None:
            return {
                "ok": False,
                "error": "invalid_argument",
                "reason": "merchant caller has no store_id",
            }
        scope_kwargs = {"store_id": ctx.store_id}
    elif ctx.role == "support":
        scope_kwargs = {"all_orders": True}
    else:
        return {
            "ok": False,
            "error": "invalid_argument",
            "reason": f"unsupported role '{ctx.role}'",
        }

    query_lower = query.lower()
    with db.connection() as conn:
        candidates = db.list_order_search_candidates(conn, **scope_kwargs)
        titles_by_product_id = {p.id: p.title for p in db.list_products(conn)}

    # Token-overlap matching: a product matches when any word of its title
    # (case-insensitive) appears in the query, so "earmuffs I bought last
    # week" matches a product titled "Wool Earmuffs". Candidates already
    # arrive newest-first (order ID descending on ties), so filtering
    # preserves that order; stop once we have five.
    matches = []
    for order in candidates:
        title_tokens = titles_by_product_id.get(order.product_id, "").lower().split()
        if title_tokens and any(token in query_lower for token in title_tokens):
            matches.append(order)
            if len(matches) == 5:
                break

    return {"ok": True, "orders": [o.to_public_dict() for o in matches]}


def check_return_eligibility(ctx: AuthContext, order_id: int) -> dict[str, Any]:
    """Explain whether an order can still be returned/refunded. Risk tier: read.

    This is a student-added tool (not part of the five HW1 holes). It fills a
    gap `get_order` leaves open: `get_order` reports a bare `refund_eligible`
    boolean, but never the numbers or the policy behind it, so the model is
    left to compute "how many days are left" itself from `delivered_at` --
    exactly the kind of date arithmetic a model can get wrong. This tool
    hands back the effective return window (the store's override if it has
    one, else the platform default), the day count, and which policy id
    governs, so the agent can cite real numbers instead of inventing them
    (SPEC RESP-1, RESP-3).

    It deliberately does NOT recompute the eligibility boolean itself. The
    boolean is trusted from `order.refund_eligible`, which is already stamped
    by the same oracle function this tool imports for the day-count math
    (seed.eligibility.effective_return_window_days), so there is exactly one
    source of truth for what counts as eligible.

    Access rules mirror `get_order`: the caller must be authorized to view
    the order (agent.auth.can_view_order), checked before any information
    about the order is returned.

    Args:
        ctx: The caller's auth context.
        order_id: The order to check.

    Returns:
        If no order has this id: {"ok": False, "error": "not_found",
        "reason": ...}.
        If the caller may not view the order: agent.auth.permission_denied(...).
        On success: {"ok": True, "order_id": order_id, "eligible": bool,
        "status": str, "return_window_days": int, "policy_id": str,
        "delivered_at": str | None, "days_since_delivery": int | None,
        "days_remaining": int | None}. The last three fields are None when
        the order has never been delivered (the return window has not
        started), in which case "eligible" is False and "reason" explains why.
    """
    facts = load_facts()
    with db.connection() as conn:
        order = db.get_order(conn, order_id)
        if order is None:
            return {"ok": False, "error": "not_found", "reason": f"no order #{order_id}"}
        if not can_view_order(ctx, order.user_id, order.store_id):
            return permission_denied(
                f"role '{ctx.role}' (user {ctx.user_id}) may not view order #{order_id}"
            )
        store = db.get_store(conn, order.store_id)
        today = db.world_asof(conn)

    store_override = store.return_window_days_override if store else None
    window_days = effective_return_window_days(facts["return_window_days"], store_override)
    policy_id = f"store-{store.slug}-policy" if store_override is not None else "cw-returns"

    if order.delivered_at is None:
        return {
            "ok": True,
            "order_id": order_id,
            "eligible": False,
            "status": order.status,
            "return_window_days": window_days,
            "policy_id": policy_id,
            "delivered_at": None,
            "days_since_delivery": None,
            "days_remaining": None,
            "reason": (
                f"order #{order_id} has not been delivered yet (status "
                f"'{order.status}'); the return window starts at delivery"
            ),
        }

    days_since_delivery = (today - order.delivered_at).days
    return {
        "ok": True,
        "order_id": order_id,
        "eligible": order.refund_eligible,
        "status": order.status,
        "return_window_days": window_days,
        "policy_id": policy_id,
        "delivered_at": order.delivered_at.isoformat(),
        "days_since_delivery": days_since_delivery,
        "days_remaining": window_days - days_since_delivery,
    }


def list_refunds(ctx: AuthContext, order_id: int) -> dict[str, Any]:
    """List every refund record filed against an order. Risk tier: read.

    Second student-added tool, found while examining Part B behavior: a
    support agent asked to check on a customer's refund got `get_order`,
    which has no refund information at all -- no field on the order even
    changes for a queued_for_approval refund (only an auto_approved one sets
    status to "refunded"). The agent had no way to see that a refund had
    already been filed, and answered "no refund on file" when three were
    actually queued. This tool gives the agent the missing visibility
    instead of leaving it to infer an answer it cannot actually see.

    Access rules mirror `get_order`: the caller must be authorized to view
    the order (agent.auth.can_view_order), checked before any refund
    information is returned.

    Args:
        ctx: The caller's auth context.
        order_id: The order whose refund history to list.

    Returns:
        If no order has this id: {"ok": False, "error": "not_found",
        "reason": ...}.
        If the caller may not view the order: agent.auth.permission_denied(...).
        On success: {"ok": True, "order_id": order_id, "refunds": [...],
        "count": <len(refunds)>}, newest first. Each refund is
        {"refund_id": int, "amount_usd": float, "status": str, "reason": str,
        "created_at": str}. No refunds is still a success: an empty list.
    """
    with db.connection() as conn:
        order = db.get_order(conn, order_id)
        if order is None:
            return {"ok": False, "error": "not_found", "reason": f"no order #{order_id}"}
        if not can_view_order(ctx, order.user_id, order.store_id):
            return permission_denied(
                f"role '{ctx.role}' (user {ctx.user_id}) may not view order #{order_id}"
            )
        rows = db.list_refunds_for_order(conn, order_id)

    refunds = [
        {
            "refund_id": row["id"],
            "amount_usd": row["amount_cents"] / 100,
            "status": row["status"],
            "reason": row["reason"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]
    return {"ok": True, "order_id": order_id, "refunds": refunds, "count": len(refunds)}
