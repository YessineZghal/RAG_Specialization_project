"""Load a specific agent class directly from its file path, bypassing
`sys.path` + `import agent` entirely.

Every specialized agent in this level is named `agent.py` by convention
(`retrieval-agent/agent.py`, `sql-agent/agent.py`, `graph-agent/agent.py`,
...) — deliberately, matching the plan this repo follows. That means the
usual "insert the folder, `import agent`" trick used for other hyphenated
folders throughout this repo would break the moment *two* agent modules
need to be imported into the same process (e.g. the Research Agent needs
both `retrieval-agent` and `graph-agent`): whichever loads first wins the
`agent` slot in `sys.modules`, and the second import silently gets the
wrong module. `importlib.util.spec_from_file_location` with a unique
synthetic module name per folder sidesteps the collision entirely,
regardless of how many agent folders get loaded in one process.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from .config import settings


def load_agent_class(folder_name: str, class_name: str):
    file_path = settings.level_dir / folder_name / "agent.py"
    module_name = f"_agent_module_{folder_name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)
