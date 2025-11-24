#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast Pipeline - Utilities Module
Shared functions for logging, LLM calls, JSON extraction, error handling.
Copied patterns from multileg_bot_v3.py for consistency.
"""

import os
import json
import re
import time
import requests
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

# ========================================
# CONFIGURATION
# ========================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BACKUP_API_KEY = os.getenv("OPENROUTER_BACKUP_API_KEY", "").strip()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Retry settings (from multileg_bot_v3.py)
MAX_RETRIES = 2
RETRY_DELAYS = [5, 15, 45, 135]
TIMEOUT_PT1_RESEARCH = 600
TIMEOUT_PT2_FORECAST = 300
MAX_TOTAL_TIME_PT1 = 900
MAX_TOTAL_TIME_PT2 = 900
MIN_PT1_JSON_LENGTH = 500

# Output management
_CURRENT_QUESTION_ID: Optional[str] = None
_CURRENT_OUTPUT_DIR: str = "out"

# ========================================
# OUTPUT DIRECTORY MANAGEMENT
# ========================================

def set_current_question(question_id: str) -> None:
    """Set the current question being processed and create its output directory"""
    global _CURRENT_QUESTION_ID, _CURRENT_OUTPUT_DIR
    _CURRENT_QUESTION_ID = question_id
    _CURRENT_OUTPUT_DIR = f"out/Q_{question_id}"
    try:
        os.makedirs(_CURRENT_OUTPUT_DIR, exist_ok=True)
        os.makedirs("out", exist_ok=True)
        log(f"[OUTPUT] Question {question_id} → {_CURRENT_OUTPUT_DIR}")
    except Exception as e:
        log(f"[WARN] Failed to create output directory: {e}")
        _CURRENT_OUTPUT_DIR = "out"

def get_output_path(filename: str) -> str:
    """Get full path for output file in current question's directory"""
    return os.path.join(_CURRENT_OUTPUT_DIR, filename)

def get_current_question_id() -> Optional[str]:
    """Get the current question ID being processed"""
    return _CURRENT_QUESTION_ID

# ========================================
# LOGGING (copied from multileg_bot_v3.py)
# ========================================

def ts() -> str:
    """Timestamp for logging"""
    return datetime.now(timezone.utc).strftime("[%Y-%m-%d %H:%M:%S]")

def log(msg: str) -> None:
    """Standard log message"""
    print(f"{ts()} {msg}", flush=True)

def log_progress(msg: str) -> None:
    """Special progress logging that stands out"""
    print(f"\n{ts()} {'='*60}", flush=True)
    print(f"{ts()} {msg}", flush=True)
    print(f"{ts()} {'='*60}\n", flush=True)

def save_response(filename: str, content: str) -> None:
    """Save response to current question's output directory"""
    try:
        filepath = get_output_path(filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        log(f"[SAVED] {filepath} ({len(content)} chars)")
    except Exception as e:
        log(f"[WARN] Failed to save {filename}: {e}")

# ========================================
# ERROR DETECTION (from multileg_bot_v3.py)
# ========================================

def is_retryable_error(response_text: str) -> bool:
    """Check if an error response is retryable"""
    if response_text is None:
        return True
    rt = str(response_text).strip()
    if not rt or len(rt) < 20:
        return True
    retryable_patterns = [
        "TIMEOUT", "EXCEPTION", "HTTP_429", "HTTP_500", "HTTP_502", 
        "HTTP_503", "HTTP_504", "INVALID_JSON", "EMPTY_CONTENT", 
        "NO_CHOICES", "ERROR PROCESSING STREAM", "RETRYABLE"
    ]
    upper = rt.upper()
    return any(p in upper for p in retryable_patterns)

def is_not_applicable_status(status_val: str) -> bool:
    """Check if status indicates N/A or not applicable"""
    if not status_val:
        return False
    s = str(status_val).strip().upper()
    return s in {"N/A", "NOT_APPLICABLE"} or s.endswith("_NOT_APPLICABLE")

# ========================================
# LLM CALLS (from multileg_bot_v3.py)
# ========================================

def _needs_backup_fallback(model: str) -> bool:
    """Check if a model needs backup fallback (Claude or Gemini)"""
    model_lower = model.lower()
    return model_lower.startswith("anthropic/") or model_lower.startswith("google/")

def _is_provider_error(error_text: str) -> bool:
    """Check if error is due to provider availability (404 or 'no allowed providers')"""
    error_lower = error_text.lower()
    return '404' in error_lower or 'no allowed providers' in error_lower

def call_llm(
    model: str,
    system_prompt: str,
    user_payload: str,
    temperature: float = 0.2,
    max_tokens: int = 16000,
    timeout: int = 180
) -> str:
    """
    Call OpenRouter LLM with retry logic.
    For Claude/Gemini models, includes backup OpenRouter fallback if primary fails with provider error.
    Returns response text or error JSON string.
    """
    needs_backup = _needs_backup_fallback(model)
    
    # Build headers for primary key
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    # Try primary OpenRouter
    try:
        log(f"[LLM] Calling {model} via OpenRouter (PRIMARY)")
        resp = requests.post(OPENROUTER_URL, headers=headers, json=body, timeout=timeout)
        
        if resp.status_code != 200:
            error_text = resp.text[:500]
            log(f"[ERROR] HTTP {resp.status_code}: {error_text}")
            
            # Check if it's a provider error and we have backup key (only for Claude/Gemini)
            if needs_backup and _is_provider_error(error_text) and OPENROUTER_BACKUP_API_KEY:
                log(f"[LLM] Primary OpenRouter failed (provider unavailable) for {model}")
                log(f"[LLM] 🔄 Switching to BACKUP OpenRouter")
                
                # Try backup key
                backup_headers = {
                    "Authorization": f"Bearer {OPENROUTER_BACKUP_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                try:
                    backup_resp = requests.post(OPENROUTER_URL, headers=backup_headers, json=body, timeout=timeout)
                    
                    if backup_resp.status_code != 200:
                        backup_error = backup_resp.text[:500]
                        log(f"[ERROR] BACKUP also failed - HTTP {backup_resp.status_code}: {backup_error}")
                        return f'{{"status":"HTTP_{backup_resp.status_code}","note":"{backup_error}","retryable":true}}'
                    
                    backup_data = backup_resp.json()
                    choices = backup_data.get("choices", [])
                    if not choices:
                        return '{"status":"NO_CHOICES","retryable":true}'
                    
                    content = choices[0].get("message", {}).get("content", "")
                    if not content:
                        return '{"status":"EMPTY_CONTENT","retryable":true}'
                    
                    log(f"[LLM] ✅ BACKUP OpenRouter succeeded")
                    return content
                    
                except requests.exceptions.Timeout:
                    return f'{{"status":"TIMEOUT","note":"Backup request exceeded {timeout}s","retryable":true}}'
                except Exception as backup_e:
                    log(f"[ERROR] BACKUP OpenRouter exception: {backup_e}")
                    return f'{{"status":"EXCEPTION","note":"Backup failed: {str(backup_e)[:200]}","retryable":true}}'
            
            # Not a provider error or no backup available - return error
            return f'{{"status":"HTTP_{resp.status_code}","note":"{error_text}","retryable":true}}'
        
        # Success with primary
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return '{"status":"NO_CHOICES","retryable":true}'
        
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            return '{"status":"EMPTY_CONTENT","retryable":true}'
        
        return content
    
    except requests.exceptions.Timeout:
        return f'{{"status":"TIMEOUT","note":"Request exceeded {timeout}s","retryable":true}}'
    except Exception as e:
        return f'{{"status":"EXCEPTION","note":"{str(e)[:200]}","retryable":true}}'

# ========================================
# JSON EXTRACTION (from multileg_bot_v3.py)
# ========================================

def extract_json_from_position(text: str, start_pos: int) -> Optional[str]:
    """Extract JSON object starting from a specific position"""
    depth = 0
    in_string = False
    escape = False
    
    for i in range(start_pos, len(text)):
        ch = text[i]
        
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue
        
        if ch == '"':
            in_string = True
            continue
        
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
    
    return None

def extract_json_blocks(text: str, leg_name: str = "") -> List[str]:
    """
    Extract JSON blocks from LLM response.
    Tries multiple strategies:
    1. Named JSON blocks (e.g., "KM_RESULTS: {...}")
    2. Code blocks (```json ... ```)
    3. Raw JSON objects
    """
    blocks = []
    
    # Strategy 1: Named JSON blocks
    pattern = r'\b([A-Z_][A-Z_0-9]*)\s*:\s*\{'
    for match in re.finditer(pattern, text):
        brace_pos = match.end() - 1
        json_str = extract_json_from_position(text, brace_pos)
        if json_str:
            blocks.append(json_str)
    
    # Strategy 2: Code blocks
    if not blocks:
        code_block_pattern = r'```(?:json)?\s*(\{[^`]+\})\s*```'
        for m in re.finditer(code_block_pattern, text, re.DOTALL):
            blocks.append(m.group(1).strip())
    
    # Strategy 3: Raw JSON objects
    if not blocks:
        depth = 0
        in_string = False
        escape = False
        start = -1
        
        for i, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            
            if ch == '"':
                in_string = True
                continue
            
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        blocks.append(text[start:i+1])
                        start = -1
    
    # Sort by length (longest first - usually the most complete)
    blocks.sort(key=len, reverse=True)
    return blocks

# ========================================
# VALIDATION
# ========================================

def is_valid_pt1_envelope(obj: dict, json_str: str, leg_name: str) -> bool:
    """Validate PT1 research output"""
    if not isinstance(obj, dict) or not obj:
        return False
    
    # Check if it's a valid N/A status
    if is_not_applicable_status(obj.get("status", "")):
        return True
    
    # Check minimum length
    if len(json_str) < MIN_PT1_JSON_LENGTH:
        return False
    
    return True

def validate_pt2_result(result_envelope: dict, result_key: str, leg_name: str) -> bool:
    """Validate PT2 forecast output"""
    if not isinstance(result_envelope, dict):
        return False
    
    if result_key not in result_envelope:
        return False
    
    result_data = result_envelope[result_key]
    if not isinstance(result_data, dict):
        return False
    
    # Check if it's N/A
    if is_not_applicable_status(result_data.get("status", "")):
        return False
    
    # Check for at least one branch (A, B, C, or D)
    branch_keys = [k for k in result_data.keys() if k in ['A', 'B', 'C', 'D']]
    return len(branch_keys) > 0

def extract_pt2_results(candidates: List[str], result_key: str, leg_name: str) -> Optional[dict]:
    """
    Extract PT2 results from candidate JSON blocks.
    Handles various response formats.
    """
    for raw in candidates:
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict):
                continue
            
            # Direct match
            if result_key in obj:
                return obj
            
            # Wrapped in common keys
            for wrapper_key in ['results', 'result', 'data', 'output']:
                if wrapper_key in obj and isinstance(obj[wrapper_key], dict):
                    if result_key in obj[wrapper_key]:
                        return obj[wrapper_key]
            
            # Single branch response
            if len(obj) == 1:
                branch_key = list(obj.keys())[0]
                if branch_key in ['A', 'B', 'C', 'D']:
                    return {result_key: obj}
            
            # Direct branch data
            if any(k in obj for k in ['P_YES', 'P_NO', 'MEAN', 'SD', 'CANDIDATE_PROBS', 'DATE_PROBS']):
                return {result_key: {'A': obj}}
        
        except Exception:
            continue
    
    return None

# ========================================
# NORMALIZATION
# ========================================

def normalize_branch_result(branch_data: dict) -> dict:
    """Normalize a single branch result to standard format"""
    if not isinstance(branch_data, dict):
        return branch_data
    
    normalized = {}
    for key, value in branch_data.items():
        key_upper = key.upper()
        
        if key_upper in ["MEAN", "P_YES", "P_NO", "P"]:
            normalized[key_upper] = value
        elif key_upper == "SD":
            normalized["SD"] = value
        elif key_upper in ["INTERVAL", "CENTRAL_INTERVAL", "CONFIDENCE_INTERVAL"]:
            if isinstance(value, dict):
                lo = value.get("lo") or value.get("lower") or value.get("min")
                hi = value.get("hi") or value.get("upper") or value.get("max")
                if lo is not None and hi is not None:
                    normalized["INTERVAL"] = [lo, hi]
            elif isinstance(value, list) and len(value) >= 2:
                normalized["INTERVAL"] = value[:2]
        elif key_upper in ["CANDIDATE_PROBS", "DATE_PROBS", "OPTIONS"]:
            normalized[key_upper] = value
        else:
            normalized[key_upper] = value
    
    return normalized

def normalize_leg_results(leg_results: dict, leg_name: str) -> dict:
    """Normalize leg results to standard format"""
    if not isinstance(leg_results, dict):
        return leg_results
    
    result_key = f"{leg_name}_RESULTS"
    if result_key not in leg_results:
        return leg_results
    
    result_data = leg_results[result_key]
    if not isinstance(result_data, dict):
        return leg_results
    
    # Normalize each branch
    normalized_results = {}
    for branch_key, branch_data in result_data.items():
        if branch_key in ['A', 'B', 'C', 'D']:
            normalized_results[branch_key] = normalize_branch_result(branch_data)
        else:
            normalized_results[branch_key] = branch_data
    
    # Create normalized envelope
    normalized_envelope = leg_results.copy()
    normalized_envelope[result_key] = normalized_results
    return normalized_envelope

# ========================================
# QUESTION AUGMENTATION
# ========================================

def augment_question_object(q: dict) -> dict:
    """
    Augment question object with normalized fields.
    Ensures consistent format across all methods.
    """
    q2 = dict(q)
    
    # Normalize question type
    qt_raw = str(q2.get("question_type", "")).lower()
    if qt_raw in {"multi", "mc", "multiple-choice", "multiple choice", "categorical"}:
        q2["question_type"] = "multiple_choice"
    
    # Normalize discrete to numeric (they're handled identically)
    if qt_raw == "discrete":
        q2["question_type"] = "numeric"
    
    # Multiple choice: ensure options are present
    if qt_raw == "multiple_choice":
        options = q2.get("options") or q2.get("mc_options") or []
        if options:
            q2["options"] = options
            q2["mc_options"] = options
    
    # Numeric: ensure numeric_range is present
    if qt_raw in ("numeric", "discrete") and "range" in q2 and "numeric_range" not in q2:
        q2["numeric_range"] = q2["range"]
    
    # Date: ensure date_range is present
    if qt_raw == "date" and "range" in q2 and "date_range" not in q2:
        q2["date_range"] = q2["range"]
    
    return q2

# ========================================
# STATE MANAGEMENT
# ========================================

def save_state(state: dict, output_dir: str = "out") -> None:
    """Save state to JSON file"""
    try:
        state_path = os.path.join(output_dir, "state.json")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        log(f"[STATE] Saved to {state_path}")
    except Exception as e:
        log(f"[WARN] Failed to save state: {e}")

def load_state(output_dir: str = "out") -> Optional[dict]:
    """Load state from JSON file"""
    try:
        state_path = os.path.join(output_dir, "state.json")
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            log(f"[STATE] Loaded from {state_path}")
            return state
    except Exception as e:
        log(f"[WARN] Failed to load state: {e}")
    return None

# ========================================
# END OF UTILS
# ========================================
