#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast Pipeline - Stage 3: Combiner (PUBLIC STUB)

Open-source-safe version that wraps the available inputs into 
a simple forecast object that Stage 4 (Veo/ElevenLabs) can consume.

"""

import json
from typing import Dict, Optional
from utils import log, log_progress, save_response


def combine_forecasts(
    qobj: dict,
    panshul_result: Optional[dict],
    method_results: Dict[str, dict],
    evidence_packet: dict,
) -> dict:
    """
    Public stub combiner.

    This just packages inputs into a simple structure with metadata.
    Replace with your own logic in a private version.
    """
    q_id = qobj.get("question_id", "unknown")

    # Very lightweight "combined" object – no real math here.
    combined = {
        "question": qobj,
        "inputs": {
            "panshul_result": panshul_result,
            "method_results": method_results,
            "evidence": evidence_packet,
        },
        "combined_forecast": {
            "note": (
                "Public stub combiner – no proprietary weighting or math. "
                "Implement your own combination logic here in a private fork."
            )
        },
        "metadata": {
            "question_id": q_id,
            "num_methods": len(method_results) if method_results else 0,
            "methods_used": list(method_results.keys()) if method_results else [],
            "simple_average": None,  # Left blank in the stub
        },
    }

    # Save a copy for inspection
    save_response("final_payload.json", json.dumps(combined, indent=2))
    return combined


def run_stage3(
    qobj: dict,
    panshul_result: Optional[dict],
    method_results: Dict[str, dict],
    evidence_packet_full: dict,
) -> Optional[dict]:
    """
    Public stub entrypoint for Stage 3.

    Args:
        qobj: Question object
        panshul_result: Panshul forecast or None
        method_results: Dict of method results (may be empty in public version)
        evidence_packet_full: Full OUTPUT_B_EVIDENCE_PACKET (all questions)

    Returns:
        A minimal forecast object suitable for Stage 4, or None on failure.
    """
    q_id = qobj.get("question_id", "unknown")
    log_progress(f"🎯 [PUBLIC] COMBINER STUB FOR Q {q_id}")

    # Extract evidence for THIS question only
    evidence_for_this_q = evidence_packet_full.get(q_id, {})
    if not evidence_for_this_q:
        log(f"[STAGE3] ⚠️ No evidence found for Q {q_id} in evidence packet")
        evidence_for_this_q = {
            "question_text": qobj.get("question_text", qobj.get("title", "")),
            "consequences": {
                "near_term": "No evidence available in public stub.",
                "knock_ons": "No evidence available in public stub.",
            },
        }

    result = combine_forecasts(qobj, panshul_result, method_results, evidence_for_this_q)
    log_progress(f"✅ [PUBLIC] COMBINER STUB COMPLETE FOR Q {q_id}")
    return result
