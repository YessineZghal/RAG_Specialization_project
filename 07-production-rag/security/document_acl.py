"""Document-level access control — enforced as a **pre-filter at the
vector database level** (Qdrant's `Filter`/`MatchAny`, see
`retrieval-infrastructure/qdrant.py`), not as a post-hoc check on results
already returned. That distinction matters: post-filtering can leak a
restricted document's *existence* (and let it silently starve a Top-K
result set), while pre-filtering never retrieves it in the first place.
"""

from __future__ import annotations


class DocumentACL:
    def __init__(self, doc_owners: dict[str, set[str]]) -> None:
        """`doc_owners`: doc_id -> set of user_ids allowed to see it.
        A doc_id with no entry is treated as public (visible to everyone).
        """
        self.doc_owners = doc_owners

    def allowed_doc_ids(self, user_id: str, candidate_doc_ids: list[str]) -> set[str]:
        return {
            doc_id
            for doc_id in candidate_doc_ids
            if doc_id not in self.doc_owners or user_id in self.doc_owners[doc_id]
        }

    def can_view(self, user_id: str, doc_id: str) -> bool:
        owners = self.doc_owners.get(doc_id)
        return owners is None or user_id in owners
