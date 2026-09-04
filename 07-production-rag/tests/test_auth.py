"""security/auth.py -- API key verification."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from security.auth import AuthError, verify_admin_key, verify_api_key
from production_common.config import settings


def test_verify_api_key_accepts_the_configured_key():
    assert verify_api_key(settings.api_key) == settings.api_key


def test_verify_api_key_rejects_wrong_key():
    with pytest.raises(AuthError):
        verify_api_key("not-the-real-key")


def test_verify_api_key_rejects_missing_key():
    with pytest.raises(AuthError):
        verify_api_key(None)
    with pytest.raises(AuthError):
        verify_api_key("")


def test_verify_admin_key_accepts_the_configured_admin_key():
    assert verify_admin_key(settings.admin_api_key) == "admin"


def test_verify_admin_key_rejects_the_regular_api_key():
    # The whole point of a separate admin key: a valid *regular* key must
    # never grant admin access, even though both are "a valid API key."
    with pytest.raises(AuthError):
        verify_admin_key(settings.api_key)


def test_verify_admin_key_rejects_wrong_key():
    with pytest.raises(AuthError):
        verify_admin_key("not-the-real-admin-key")


def test_verify_admin_key_rejects_missing_key():
    with pytest.raises(AuthError):
        verify_admin_key(None)
    with pytest.raises(AuthError):
        verify_admin_key("")
