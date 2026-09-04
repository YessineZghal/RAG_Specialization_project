"""retrieval-infrastructure/qdrant.py's `_point_id` -- a regression test
for a real bug: point ids used to come from each `upsert()` call's own
`enumerate()` position, so a second, single-document call (as
`api/routes.py`'s `/admin/ingest` now makes) would always get id 0,
silently overwriting whichever document happened to hold that id from
the very first bulk load. Pure logic, no live Qdrant required.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "retrieval-infrastructure"))
from qdrant import _point_id  # noqa: E402


def test_point_id_is_deterministic_for_the_same_doc_id():
    assert _point_id("doc-123") == _point_id("doc-123")


def test_point_id_differs_for_different_doc_ids():
    assert _point_id("doc-123") != _point_id("doc-456")


def test_point_id_never_collapses_everything_to_the_same_value():
    # The exact real failure mode: every one of these used to become
    # position 0 if it were the *first* item in its own upsert() call.
    ids = [_point_id(f"doc-{i}") for i in range(50)]
    assert len(set(ids)) == 50


def test_point_id_is_a_valid_uuid_string():
    import uuid

    # Qdrant point ids must be an unsigned int or a UUID -- confirm the
    # generated value actually parses as one.
    parsed = uuid.UUID(_point_id("some-doc-id"))
    assert str(parsed) == _point_id("some-doc-id")
