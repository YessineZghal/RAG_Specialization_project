"""Role-based permissions — separate from *authentication* (who are you)
and *document ACLs* (which documents can you see): this is about which
*actions* a given role may perform at all.
"""

from __future__ import annotations

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin": {"query", "ingest", "delete", "view_metrics"},
    "user": {"query"},
    "readonly": {"query", "view_metrics"},
}


class PermissionDeniedError(Exception):
    pass


def require_permission(role: str, action: str) -> None:
    allowed = ROLE_PERMISSIONS.get(role, set())
    if action not in allowed:
        raise PermissionDeniedError(f"Role {role!r} is not permitted to perform {action!r}.")
