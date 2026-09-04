"""security/permissions.py -- role-based action permissions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from security.permissions import PermissionDeniedError, require_permission  # noqa: E402


def test_admin_can_do_everything_a_user_can():
    require_permission("admin", "query")
    require_permission("admin", "ingest")
    require_permission("admin", "delete")
    require_permission("admin", "view_metrics")


def test_plain_user_can_only_query():
    require_permission("user", "query")
    with pytest.raises(PermissionDeniedError):
        require_permission("user", "delete")


def test_readonly_can_view_metrics_but_not_ingest():
    require_permission("readonly", "view_metrics")
    with pytest.raises(PermissionDeniedError):
        require_permission("readonly", "ingest")


def test_unknown_role_is_denied_everything():
    with pytest.raises(PermissionDeniedError):
        require_permission("guest", "query")


def test_permission_denied_error_does_not_shadow_the_python_builtin():
    # A real near-miss during this level's build: an earlier version of
    # permissions.py defined `class PermissionError(Exception)`, silently
    # shadowing the builtin of the same name. Renamed to
    # PermissionDeniedError before it could cause a real bug -- this test
    # pins that it stays a *distinct* exception type, not the builtin.
    assert PermissionDeniedError is not PermissionError
    assert issubclass(PermissionDeniedError, Exception)
