"""Named `production_eval`, not `evaluation` -- Level 2 (`02-advanced-rag/evaluation/`)
already owns that name as a real Python package (has its own `__init__.py`).
When every level's directory sits on `sys.path` at once (a full-repo
`pytest` run), a *regular* package with `__init__.py` wins the `sys.modules`
name outright over a same-named package elsewhere, regardless of path
order or which one is actually needed -- a real collision this level hit
by actually running the combined suite, same root cause as the
`modular_common`/`common` lesson from Level 3, just against a different
shared folder name this time.
"""
