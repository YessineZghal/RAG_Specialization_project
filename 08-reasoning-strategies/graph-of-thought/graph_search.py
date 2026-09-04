"""Graph-of-Thoughts' search loop -- generalizes Tree-of-Thought's beam
search (`tree-of-thought/tree_search.py`) with the one operation a tree
cannot represent: **aggregation**. Where Tree-of-Thought only ever
branches and prunes, this can also merge two or more promising branches
into one new, synthesized node that has both as parents -- a real graph,
built with `thought_graph.py`, not a re-implementation of the tree search
under a different name.

Reuses `tree-of-thought/thought_generator.py` and `state_evaluator.py`
directly rather than duplicating them -- branching and scoring are the
same problem in both strategies; aggregation is the only genuinely new
piece Graph-of-Thoughts adds.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thought_graph import ThoughtGraph  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tree-of-thought"))
from state_evaluator import evaluate_state  # noqa: E402
from thought_generator import generate_thoughts  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reasoning_common.answer_parsing import parse_yes_no  # noqa: E402
from reasoning_common.llm import OllamaLLM  # noqa: E402

AGGREGATE_PROMPT = (
    "Context:\n{context}\n\nQuestion: {question}\n\n"
    "Two different reasoning branches were explored:\n\n"
    "Branch 1:\n{branch_a}\n\nBranch 2:\n{branch_b}\n\n"
    "Write a single synthesized reasoning step (one or two sentences) "
    "that combines whatever is useful from both branches into one "
    "coherent next step. Respond with ONLY the synthesized step."
)

FINAL_ANSWER_PROMPT = (
    "Context:\n{context}\n\nQuestion: {question}\n\n"
    "Reasoning path:\n{path}\n\n"
    "Based on this reasoning, give your final answer on its own line, in "
    "exactly this form: 'Answer: Yes' or 'Answer: No'."
)


def aggregate_thoughts(branch_a: str, branch_b: str, question: str, context: str, llm: OllamaLLM) -> str:
    prompt = AGGREGATE_PROMPT.format(context=context, question=question, branch_a=branch_a, branch_b=branch_b)
    return llm.complete(prompt).strip()


def graph_of_thought_search(
    question: str,
    context: str,
    llm: OllamaLLM | None = None,
    branching_factor: int = 3,
    max_depth: int = 2,
    beam_width: int = 2,
    score_threshold: float = 0.8,
) -> dict:
    llm = llm or OllamaLLM()
    llm_calls = 0

    graph = ThoughtGraph()
    root_id = graph.add_thought("(start)", score=1.0)
    frontier = [root_id]

    for depth in range(max_depth):
        candidates: list[int] = []
        for node_id in frontier:
            path = graph.path_to(node_id)[1:]  # drop the synthetic "(start)" placeholder
            thoughts = generate_thoughts(question, context, path, k=branching_factor, llm=llm)
            llm_calls += 1
            for thought in thoughts:
                new_path = [*path, thought]
                score = evaluate_state(question, context, new_path, llm=llm)
                llm_calls += 1
                candidates.append(graph.add_thought(thought, parents=[node_id], score=score))

        if not candidates:
            break

        candidates.sort(key=lambda nid: graph.score(nid), reverse=True)
        # Consider the top 2 candidates for aggregation *before* narrowing
        # to `beam_width` -- otherwise a beam_width of 1 would make
        # aggregation impossible by construction, defeating the point of
        # having it at all.
        aggregation_pool = candidates[:max(beam_width, 2)]

        # Aggregation: the real graph-shaped step. If at least two
        # branches survived pruning, and this is not the final round
        # (nothing left to build on afterward), merge the best two into
        # one synthesized node with *both* as parents.
        if len(aggregation_pool) >= 2 and depth < max_depth - 1:
            merged_text = aggregate_thoughts(
                graph.text(aggregation_pool[0]), graph.text(aggregation_pool[1]), question, context, llm
            )
            llm_calls += 1
            merged_score = evaluate_state(question, context, [merged_text], llm=llm)
            llm_calls += 1
            merged_id = graph.add_thought(merged_text, parents=[aggregation_pool[0], aggregation_pool[1]], score=merged_score)
            pool = sorted([merged_id, *aggregation_pool], key=lambda nid: graph.score(nid), reverse=True)
        else:
            pool = aggregation_pool

        frontier = pool[:beam_width]

        if graph.score(frontier[0]) >= score_threshold:
            break

    best_id = max(frontier, key=lambda nid: graph.score(nid))
    best_path = graph.path_to(best_id)[1:]

    final_prompt = FINAL_ANSWER_PROMPT.format(
        context=context, question=question, path="\n".join(best_path) or "(no reasoning path found)"
    )
    raw = llm.complete(final_prompt)
    llm_calls += 1

    return {
        "answer": parse_yes_no(raw),
        "best_path": best_path,
        "best_score": graph.score(best_id),
        "reasoning": raw,
        "llm_calls": llm_calls,
        "graph_size": len(graph),
        "graph": graph,  # exposed for inspection/visualization -- see notebooks/03_graph_of_thought.ipynb
    }
