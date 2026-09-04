"""Named `reasoning_eval`, not `evaluation` -- Level 2 (`02-advanced-rag/evaluation/`)
already owns that name as a real Python package (it has its own
`__init__.py`). When every level's directory sits on `sys.path` at once
(this repo's root `pyproject.toml` puts every level's directory on
`pythonpath` unconditionally, for every `pytest` invocation, not just a
combined one), a *regular* package with `__init__.py` wins the
`sys.modules` name outright over a same-named package elsewhere,
regardless of path order or which one a given test actually needs --
confirmed here immediately, the moment this level's own tests were run
for the first time (`ModuleNotFoundError: No module named
'evaluation.cost_tracker'`), the same root cause already documented in
[Level 7](../../07-production-rag/production_eval/__init__.py) after it
hit the identical collision.
"""
