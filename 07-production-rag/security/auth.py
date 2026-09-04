"""API key authentication — the minimum viable gate before anything else
in this level's security stack applies. FastAPI's dependency-injection
system wires this in per-route (see `api/routes.py`), not as a global
middleware, so unauthenticated endpoints (health, metrics) stay reachable
for monitoring infrastructure.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from production_common.config import settings


class AuthError(Exception):
    pass


def verify_api_key(provided_key: str | None) -> str:
    """Return a user id on success (here, just the key itself stands in
    for a user id -- a real system would look up a user record). Raise
    `AuthError` on failure.
    """
    if not provided_key:
        raise AuthError("Missing API key.")
    if provided_key != settings.api_key:
        raise AuthError("Invalid API key.")
    return provided_key


def verify_admin_key(provided_key: str | None) -> str:
    """Same shape as `verify_api_key`, checked against a *separate* admin
    key (`settings.admin_api_key`) -- a regular caller's key is never
    valid here, even by accident, because it is compared against a
    different setting entirely. Returns the fixed role name `"admin"`,
    which `api/routes.py`'s `/admin/ingest` route then passes to
    `permissions.require_permission()` -- this is what actually
    exercises that function and the `"ingest"` action in real, running
    code for the first time; every prior use of both was in tests only.
    """
    if not provided_key:
        raise AuthError("Missing admin API key.")
    if provided_key != settings.admin_api_key:
        raise AuthError("Invalid admin API key.")
    return "admin"
