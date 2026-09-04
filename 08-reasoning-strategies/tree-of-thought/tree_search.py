"""Tree-of-Thought's search loop -- the actual "tree" in Tree-of-Thought.
Ties `thought_generator.py` (branch) and `state_evaluator.py` (score)
together into a real beam search: at each depth, every surviving branch
proposes several next steps, every new branch gets scored, and only the
top `beam_width` branches survive to the next depth -- the rest are
pruned, which is this level's concrete form of "backtracking away from
unpromising branches." A `max_depth` cap and an early-stop once a branch
scores above `score_threshold` both exist for the same reason Level 5's
agent loop had a max-steps fallback: an unbounded search is a real risk,
not a hypothetical one.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from state_evaluator import evaluate_state
from thought_generator import generate_thoughts

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reasoning_common.answer_parsing import parse_yes_no
from reasoning_common.llm import OllamaLLM

FINAL_ANSWER_PROMPT = (
    "Context:\n{context}\n\nQuestion: {question}\n\n"
    "Reasoning path:\n{path}\n\n"
    "Based on this reasoning, give your final answer on its own line, in "
    "exactly this form: 'Answer: Yes' or 'Answer: No'."
)


@dataclass
class TreeNode:
    path: list[str] = field(default_factory=list)
    score: float = 1.0  # the root starts at 1.0 -- nothing to be unpromising about yet
    depth: int = 0


def tree_of_thought_search(
    question: str,
    context: str,
    llm: OllamaLLM | None = None,
    branching_factor: int = 3,
    max_depth: int = 3,
    beam_width: int = 2,
    score_threshold: float = 0.8,
) -> dict:
    llm = llm or OllamaLLM()
    llm_calls = 0

    frontier = [TreeNode()]
    best_node = frontier[0]

    for _depth in range(max_depth):
        candidates: list[TreeNode] = []
        for node in frontier:
            thoughts = generate_thoughts(question, context, node.path, k=branching_factor, llm=llm)
            llm_calls += 1
            for thought in thoughts:
                new_path = [*node.path, thought]
                score = evaluate_state(question, context, new_path, llm=llm)
                llm_calls += 1
                candidates.append(TreeNode(path=new_path, score=score, depth=node.depth + 1))

        if not candidates:
            break  # every branch produced nothing usable -- stop rather than loop on empty input

        candidates.sort(key=lambda n: n.score, reverse=True)
        frontier = candidates[:beam_width]  # prune -- the rest are the backtracked-away branches
        best_node = frontier[0]

        if best_node.score >= score_threshold:
            break

    final_prompt = FINAL_ANSWER_PROMPT.format(
        context=context, question=question, path="\n".join(best_node.path) or "(no reasoning path found)"
    )
    raw = llm.complete(final_prompt)
    llm_calls += 1

    return {
        "answer": parse_yes_no(raw),
        "best_path": best_node.path,
        "best_score": best_node.score,
        "reasoning": raw,
        "llm_calls": llm_calls,
    }
