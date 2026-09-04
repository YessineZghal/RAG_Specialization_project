"""Locust load test against the running API — real questions from the
real SQuAD sample, so the load test exercises genuine retrieval +
generation (and, since the same questions repeat across simulated users,
genuine cache hits too) rather than one canned payload.

Usage:
    cd 07-production-rag
    uv run locust -f load-testing/locustfile.py --host http://127.0.0.1:8001
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from locust import HttpUser, between, task

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from production_common.dataset import prepare

_data = prepare()
_QUESTIONS = [q["question"] for q in _data.questions.values()]


class RagApiUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(8)
    def query(self) -> None:
        question = random.choice(_QUESTIONS)
        self.client.post(
            "/query",
            json={"question": question, "top_k": 5},
            headers={"x-api-key": "dev-local-key"},
            name="/query",
        )

    @task(2)
    def health(self) -> None:
        self.client.get("/health", name="/health")
