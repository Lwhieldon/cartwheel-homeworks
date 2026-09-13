"""Tests for the student-added HW1 tool: check_return_eligibility.

Not one of the five supplied contract tests -- added because Part A also
asks for a tool of the student's own, tested the same way as the required
five: a plain call into agent/tools.py, no LLM, no API keys, using the
same `world` fixture as tests/test_hw_holes.py.
"""

from __future__ import annotations

from agent import db, tools
from agent.auth import AuthContext
from agent.config import load_facts

SHOPPER_1 = AuthContext(user_id=1, role="shopper")
MERCHANT_STORE_2 = AuthContext(user_id=9002, role="merchant", store_id=2)
SUPPORT = AuthContext(user_id=9501, role="support")


def test_within_window_no_override(world: dict) -> None:
    """Order 4127: the demo store has no override, delivered inside the window."""
    facts = load_facts()
    result = tools.check_return_eligibility(SHOPPER_1, 4127)
    assert result["ok"] is True
    assert result["eligible"] is True
    assert result["policy_id"] == "cw-returns"
    assert result["return_window_days"] == facts["return_window_days"]
    assert result["days_remaining"] == result["return_window_days"] - result["days_since_delivery"]


def test_outside_window_no_override(world: dict) -> None:
    """Order 3980: same store, delivered, but past the return window."""
    result = tools.check_return_eligibility(SHOPPER_1, 3980)
    assert result["ok"] is True
    assert result["eligible"] is False
    assert result["policy_id"] == "cw-returns"
    assert result["days_remaining"] < 0


def test_store_override_changes_window_and_policy_id(world: dict) -> None:
    """A store with its own return-window override should use its own policy id."""
    with db.connection() as conn:
        store_row = conn.execute(
            "SELECT id, return_window_days_override, slug FROM stores "
            "WHERE return_window_days_override IS NOT NULL LIMIT 1"
        ).fetchone()
        store_id, override_days, slug = (
            store_row["id"],
            store_row["return_window_days_override"],
            store_row["slug"],
        )
        order_row = conn.execute(
            "SELECT id, user_id FROM orders WHERE store_id = ? AND delivered_at IS NOT NULL LIMIT 1",
            (store_id,),
        ).fetchone()
    assert order_row is not None, "expected a delivered order at an overridden store"

    ctx = AuthContext(user_id=order_row["user_id"], role="shopper")
    result = tools.check_return_eligibility(ctx, order_row["id"])
    assert result["ok"] is True
    assert result["policy_id"] == f"store-{slug}-policy"
    assert result["return_window_days"] == override_days


def test_permission_denied_matches_get_order_scope(world: dict) -> None:
    """A merchant from the wrong store is denied, and learns nothing about the order."""
    result = tools.check_return_eligibility(MERCHANT_STORE_2, 4127)
    assert result == {
        "ok": False,
        "error": "permission_denied",
        "reason": "role 'merchant' (user 9002) may not view order #4127",
    }


def test_not_found(world: dict) -> None:
    result = tools.check_return_eligibility(SHOPPER_1, 999999)
    assert result["ok"] is False
    assert result["error"] == "not_found"


def test_not_yet_delivered(world: dict) -> None:
    """No delivered_at means the return window has not started, whatever the status."""
    with db.connection() as conn:
        row = conn.execute("SELECT id, user_id FROM orders WHERE delivered_at IS NULL LIMIT 1").fetchone()
    assert row is not None, "expected at least one order that has not been delivered"

    ctx = AuthContext(user_id=row["user_id"], role="shopper")
    result = tools.check_return_eligibility(ctx, row["id"])
    assert result["ok"] is True
    assert result["eligible"] is False
    assert result["delivered_at"] is None
    assert result["days_since_delivery"] is None
    assert result["days_remaining"] is None


def test_support_can_check_any_order(world: dict) -> None:
    """Support has the same view-any-order authority here as get_order."""
    result = tools.check_return_eligibility(SUPPORT, 4127)
    assert result["ok"] is True
    assert result["eligible"] is True


# ---------------------------------------------------------------------------
# Second student-added tool: list_refunds. Added after Part B caught the
# agent asserting "no refund on file" for an order that actually had three
# queued_for_approval refunds -- get_order has no refund visibility at all.
# ---------------------------------------------------------------------------


def test_list_refunds_empty_is_still_success(world: dict) -> None:
    """An order nobody has ever filed a refund against returns an empty list, not an error."""
    result = tools.list_refunds(SHOPPER_1, 4127)
    assert result == {"ok": True, "order_id": 4127, "refunds": [], "count": 0}


def test_list_refunds_returns_inserted_refunds_newest_first(world_copy) -> None:
    """Mutates state, so this uses world_copy (its own database copy), not world."""
    with db.connection() as conn:
        first_id = db.insert_refund(
            conn, order_id=4455, amount_cents=24000, reason="first attempt",
            status="queued_for_approval", created_at="2026-07-01",
        )
        second_id = db.insert_refund(
            conn, order_id=4455, amount_cents=24000, reason="second attempt",
            status="queued_for_approval", created_at="2026-07-02",
        )

    result = tools.list_refunds(SHOPPER_1, 4455)
    assert result["ok"] is True
    assert result["count"] == 2
    # Newest first: the second insert (higher id) comes before the first.
    assert [r["refund_id"] for r in result["refunds"]] == [second_id, first_id]
    assert result["refunds"][0] == {
        "refund_id": second_id, "amount_usd": 240.0, "status": "queued_for_approval",
        "reason": "second attempt", "created_at": "2026-07-02",
    }


def test_list_refunds_permission_denied_matches_get_order_scope(world: dict) -> None:
    result = tools.list_refunds(MERCHANT_STORE_2, 4127)
    assert result == {
        "ok": False,
        "error": "permission_denied",
        "reason": "role 'merchant' (user 9002) may not view order #4127",
    }


def test_list_refunds_not_found(world: dict) -> None:
    result = tools.list_refunds(SHOPPER_1, 999999)
    assert result["ok"] is False
    assert result["error"] == "not_found"
