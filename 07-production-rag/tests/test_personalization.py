"""security/personalization.py -- reordering results per user, never
changing which documents a user is allowed to see (that is document_acl.py's
job alone)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from security.personalization import PersonalizationEngine, UserProfile

CORPUS = {
    "doc-sports": {"title": "Local team wins championship", "text": "A thrilling sports victory."},
    "doc-science": {"title": "New physics discovery", "text": "Researchers report a science breakthrough."},
    "doc-cooking": {"title": "A simple pasta recipe", "text": "Cooking instructions for dinner."},
}

# Same base retrieval scores for every test -- personalization must be the
# only thing that can change the order below.
BASE_RESULTS = [
    {"doc_id": "doc-cooking", "score": 0.80},
    {"doc_id": "doc-science", "score": 0.78},
    {"doc_id": "doc-sports", "score": 0.76},
]


def test_user_with_no_profile_gets_the_original_order_unchanged():
    engine = PersonalizationEngine(profiles={})
    result = engine.rerank("nobody", BASE_RESULTS, CORPUS)
    assert [r["doc_id"] for r in result] == ["doc-cooking", "doc-science", "doc-sports"]
    assert [r["base_score"] for r in result] == [0.80, 0.78, 0.76]


def test_user_with_an_empty_interest_set_gets_the_original_order_unchanged():
    profiles = {"alice": UserProfile(user_id="alice", interests=frozenset())}
    engine = PersonalizationEngine(profiles=profiles)
    result = engine.rerank("alice", BASE_RESULTS, CORPUS)
    assert [r["doc_id"] for r in result] == ["doc-cooking", "doc-science", "doc-sports"]


def test_two_users_get_provably_different_rankings_for_the_same_query_and_same_visible_documents():
    # This is the concrete claim this module exists to prove: the same
    # query, over the exact same (fully visible to both) documents, ranks
    # differently depending on who is asking.
    profiles = {
        "sports_fan": UserProfile(user_id="sports_fan", interests=frozenset({"sports", "championship"})),
        "scientist": UserProfile(user_id="scientist", interests=frozenset({"physics", "science"})),
    }
    engine = PersonalizationEngine(profiles=profiles)

    sports_fan_result = engine.rerank("sports_fan", BASE_RESULTS, CORPUS)
    scientist_result = engine.rerank("scientist", BASE_RESULTS, CORPUS)

    assert [r["doc_id"] for r in sports_fan_result][0] == "doc-sports"
    assert [r["doc_id"] for r in scientist_result][0] == "doc-science"
    assert [r["doc_id"] for r in sports_fan_result] != [r["doc_id"] for r in scientist_result]


def test_personalization_only_reorders_never_adds_or_removes_documents():
    profiles = {"alice": UserProfile(user_id="alice", interests=frozenset({"sports"}))}
    engine = PersonalizationEngine(profiles=profiles)

    result = engine.rerank("alice", BASE_RESULTS, CORPUS)

    assert {r["doc_id"] for r in result} == {r["doc_id"] for r in BASE_RESULTS}
    assert len(result) == len(BASE_RESULTS)


def test_boost_is_capped_at_max_boosted_matches():
    # "science" and "discovery" and "physics" all appear in doc-science's
    # text, but max_boosted_matches=2 means only 2 of those matches count.
    profile = UserProfile(
        user_id="alice",
        interests=frozenset({"physics", "science", "discovery"}),
        boost_per_match=0.10,
        max_boosted_matches=2,
    )
    engine = PersonalizationEngine(profiles={"alice": profile})

    result = engine.rerank("alice", BASE_RESULTS, CORPUS)
    science_result = next(r for r in result if r["doc_id"] == "doc-science")

    assert science_result["interest_matches"] == 2
    assert science_result["score"] == 0.78 + 2 * 0.10


def test_base_score_is_preserved_alongside_the_boosted_score():
    profiles = {"alice": UserProfile(user_id="alice", interests=frozenset({"sports"}))}
    engine = PersonalizationEngine(profiles=profiles)

    result = engine.rerank("alice", BASE_RESULTS, CORPUS)
    sports_result = next(r for r in result if r["doc_id"] == "doc-sports")

    assert sports_result["base_score"] == 0.76
    assert sports_result["score"] > sports_result["base_score"]
