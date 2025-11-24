#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast Pipeline - Method Runner (PUBLIC STUB)

Open-source-safe version.

"""

from typing import Dict
from utils import log, log_progress


def run_all_methods(qobj: dict) -> Dict[str, dict]:
    """
    Public stub for Stage 2 method runner.

    Args:
        qobj: Question object.

    Returns:
        An empty dict (no KM/BD/EX results). Downstream stages should be
        prepared to handle this (e.g. rely on Panshul or custom methods).
    """
    q_id = qobj.get("question_id", "unknown")
    log_progress(f"🔬 [PUBLIC] METHOD RUNNER STUB FOR Q {q_id}")
    log("[METHODS] Public stub: Additional legs are not included in this repo.")
    log("[METHODS] Implement methods here if forking this project.")
    return {}
