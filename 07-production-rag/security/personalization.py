"""Personalization -- ranking results differently per user, not just
deciding what a user is allowed to see.

`document_acl.py` answers "can this user see this document at all?" --
a yes/no question, the same answer for every question that user ever
asks. This module answers a different question: "given two users who can
both see the exact same documents, should the same query rank them in
the same order?" A real recommendation or search system routinely does
not -- a user who has shown interest in one topic should see documents
about it ranked higher than a user who has shown no such interest, even
when both users are asking the identical question over the identical,
fully-visible corpus.

The mechanism here is a small, keyword-based interest boost: a user
profile lists topics they have shown a real interest in (from their own
query history, in a real system -- supplied directly here for
demonstration), and any candidate document whose text matches one of
those topics gets a modest score boost before the results are
re-sorted. A user with no profile, or an empty one, sees results in
their original, unpersonalized order -- personalization only ever
reorders results a user is already allowed to see; it never grants or
removes visibility, which stays `document_acl.py`'s job alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    interests: frozenset[str] = field(default_factory=frozenset)
    # Added once per matched interest keyword, capped at max_boosted_matches
    # keywords so one document stuffed with every interest keyword cannot
    # dominate the ranking outright.
    boost_per_match: float = 0.05
    max_boosted_matches: int = 3


class PersonalizationEngine:
    def __init__(self, profiles: dict[str, UserProfile] | None = None) -> None:
        self.profiles = profiles or {}

    def rerank(
        self,
        user_id: str,
        results: list[dict],
        corpus: dict[str, dict],
    ) -> list[dict]:
        """`results`: a list of `{"doc_id": ..., "score": ...}` dicts, the
        same shape `retrieval-infrastructure/qdrant.py`'s `search()`
        already returns. Returns a new list, re-sorted by the boosted
        score, with the original score preserved under `base_score` so a
        caller can show both.

        A user with no profile (or a profile with no interests) gets
        their results back in their original order, untouched -- there is
        nothing to personalize against.
        """
        profile = self.profiles.get(user_id)
        if profile is None or not profile.interests:
            return [{**r, "base_score": r["score"]} for r in results]

        boosted = []
        for r in results:
            doc = corpus.get(r["doc_id"], {})
            haystack = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
            match_count = sum(1 for interest in profile.interests if interest.lower() in haystack)
            match_count = min(match_count, profile.max_boosted_matches)
            boosted_score = r["score"] + match_count * profile.boost_per_match
            boosted.append({**r, "base_score": r["score"], "score": boosted_score, "interest_matches": match_count})

        boosted.sort(key=lambda r: r["score"], reverse=True)
        return boosted
