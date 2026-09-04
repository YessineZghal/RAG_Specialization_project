"""Regression suite — run this level's evaluation on a fixed real
question sample, compare against a stored baseline, and fail loudly if
quality dropped. This is what "detect quality regressions before
deploying a new embedding model or prompt" (Level 7's stated success
criteria) actually looks like as code, not just a checklist item.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from production_common.config import settings  # noqa: E402

BASELINE_FILE = settings.level_dir / "production_eval" / "baseline_metrics.json"
REGRESSION_TOLERANCE = 0.05  # allow up to a 5-point drop before flagging


def save_baseline(metrics: dict) -> None:
    BASELINE_FILE.write_text(json.dumps(metrics, indent=2))


def load_baseline() -> dict | None:
    if not BASELINE_FILE.exists():
        return None
    return json.loads(BASELINE_FILE.read_text())


def check_regression(current_metrics: dict, baseline: dict | None = None) -> dict:
    baseline = baseline if baseline is not None else load_baseline()
    if baseline is None:
        return {"status": "no_baseline", "regressions": []}

    regressions = []
    for metric_name, current_value in current_metrics.items():
        baseline_value = baseline.get(metric_name)
        if baseline_value is None:
            continue
        drop = baseline_value - current_value
        if drop > REGRESSION_TOLERANCE:
            regressions.append(
                {"metric": metric_name, "baseline": baseline_value, "current": current_value, "drop": drop}
            )

    return {"status": "regression" if regressions else "ok", "regressions": regressions}


if __name__ == "__main__":
    # CLI entry point for docker/worker.Dockerfile and CI: compare the last
    # saved run (evaluation/last_run_metrics.json, produced by actually
    # running the eval -- see notebooks/03_observability_and_evaluation.ipynb)
    # against the stored baseline, and exit non-zero on a real regression so
    # a pipeline can fail the build on it.
    last_run_file = BASELINE_FILE.parent / "last_run_metrics.json"
    if not last_run_file.exists():
        print(f"No {last_run_file.name} found -- run the evaluation notebook first.")
        raise SystemExit(1)

    current = json.loads(last_run_file.read_text())
    outcome = check_regression(current)
    print(json.dumps(outcome, indent=2))
    if outcome["status"] == "regression":
        raise SystemExit(1)
