"""Authentication tests for the HW2 endpoint (Part D).

Pure Python calls into server/app.py -- no HTTP server, no Docker, no
Langfuse, no model provider key. Tracing correctness is checked separately,
against real spans, in Part E.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from server import app as server_app


@pytest.fixture(autouse=True)
def _clear_sessions(world):
    server_app._SESSIONS.clear()
    yield
    server_app._SESSIONS.clear()


def test_create_session_rejects_role_mismatch(world) -> None:
    """User 1 is seeded as a shopper; claiming "merchant" for that id must
    be refused rather than trusted, per SPEC.md AUTH-1."""
    with pytest.raises(HTTPException) as exc_info:
        server_app.create_session(server_app.SessionCreate(user_id=1, role="merchant"))
    assert exc_info.value.status_code == 403
    assert not server_app._SESSIONS


def test_token_does_not_authorize_a_different_session(world) -> None:
    """A token issued for session A must not authorize session B, even
    though both sessions are independently valid."""
    session_a = server_app.create_session(server_app.SessionCreate(user_id=1, role="shopper"))
    session_b = server_app.create_session(server_app.SessionCreate(user_id=2, role="shopper"))

    with pytest.raises(HTTPException) as exc_info:
        server_app._authorize(session_b["session_id"], f"Bearer {session_a['token']}")
    assert exc_info.value.status_code == 403
