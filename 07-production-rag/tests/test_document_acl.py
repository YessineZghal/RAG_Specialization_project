"""security/document_acl.py -- pre-filter ACL, not post-hoc filtering."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from security.document_acl import DocumentACL


def test_doc_with_no_owner_entry_is_public():
    acl = DocumentACL(doc_owners={})
    assert acl.can_view("alice", "doc1") is True
    assert acl.allowed_doc_ids("alice", ["doc1", "doc2"]) == {"doc1", "doc2"}


def test_restricted_doc_only_visible_to_its_owners():
    acl = DocumentACL(doc_owners={"secret": {"alice"}})
    assert acl.can_view("alice", "secret") is True
    assert acl.can_view("bob", "secret") is False


def test_allowed_doc_ids_filters_out_restricted_docs_for_non_owners():
    acl = DocumentACL(doc_owners={"secret": {"alice"}, "public": set()})
    # An empty owner set means nobody but... note: empty set is NOT "no
    # entry" -- it means explicitly nobody. This distinguishes "public"
    # (no key at all) from "restricted to nobody" (key present, empty set).
    allowed = acl.allowed_doc_ids("bob", ["secret", "public", "unlisted"])
    assert allowed == {"unlisted"}
