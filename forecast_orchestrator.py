#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast Pipeline - Main Orchestrator
Runs the complete 4-stage forecasting pipeline:
  Stage 1: Question Generation 
  Stage 2: Forecast Methods
  Stage 3: Combiner 
  Stage 4: Veo Scripts (from all forecasts)
"""

import os
import sys
import json
from datetime import datetime, timezone
from typing import Optional, Dict, List

# Import all stages
from utils import log, log_progress, set_current_question, save_state, load_state
from stage1_question_gen import run_stage1
from panshul_wrapper import run_panshul
from method_runner import run_all_methods
from stage3_combiner import run_stage3
from stage4_veo import run_stage4

# ========================================
# 🎯 CONFIGURATION (Read from environment with defaults)
# ========================================

# Question limits
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", "3"))

# Mode detection (explicit MODE from workflow, or auto-detect from USER_QUESTION_JSON)
MODE = os.getenv("MODE", "normal").lower()
USER_QUESTION_JSON = os.getenv("USER_QUESTION_JSON", "").strip()

# Determine if we're in user mode
if MODE == "user" or (MODE == "normal" and USER_QUESTION_JSON):
    USER_MODE = True
else:
    USER_MODE = False

# Stage toggles (read from environment, defaults to True)
ENABLE_STAGE1_QUESTION_GEN = not USER_MODE  # Disabled in user mode
ENABLE_PANSHUL_BOT = os.getenv("ENABLE_PANSHUL", "true").lower() == "true"
ENABLE_METHOD_RUNNER = os.getenv("ENABLE_METHODS", "true").lower() == "true"
ENABLE_VEO_GENERATION = os.getenv("ENABLE_VEO", "true").lower() == "true"

# Topic for Stage 1 (if not in user mode)
TOPIC = os.getenv("TOPIC", "").strip() or "Current events and trends"

# Verbose logging
VERBOSE = True

# Output directory
OUTPUT_DIR = "out"

# ========================================
# LOG CONFIGURATION ON STARTUP
# ========================================

def log_configuration():
    """Log current configuration"""
    log("=" * 80)
    log("🎯 CONFIGURATION")
    log("=" * 80)
    log(f"Mode: {'USER' if USER_MODE else 'NORMAL'}")
    log(f"Max Questions: {MAX_QUESTIONS}")
    log(f"Stage 1 (Question Gen): {'ENABLED' if ENABLE_STAGE1_QUESTION_GEN else 'DISABLED (User Mode)'}")
    log(f"Panshul Bot: {'ENABLED' if ENABLE_PANSHUL_BOT else 'DISABLED'}")
    log(f"Method Runner (KM/BD/EX): {'ENABLED' if ENABLE_METHOD_RUNNER else 'DISABLED'}")
    log(f"Veo Generation: {'ENABLED' if ENABLE_VEO_GENERATION else 'DISABLED'}")
    if not USER_MODE:
        log(f"Topic: {TOPIC}")
    else:
        log(f"User Question: {USER_QUESTION_JSON[:100]}..." if len(USER_QUESTION_JSON) > 100 else USER_QUESTION_JSON)
    log("=" * 80 + "\n")

# ========================================
# STATE MANAGEMENT
# ========================================

def initialize_state() -> dict:
    """Initialize empty state"""
    return {
        "stage": "0",
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "max_questions": MAX_QUESTIONS,
            "enable_panshul": ENABLE_PANSHUL_BOT,
            "enable_methods": ENABLE_METHOD_RUNNER,
            "enable_veo": ENABLE_VEO_GENERATION,
            "user_mode": USER_MODE,
            "topic": TOPIC if not USER_MODE else None
        },
        "stage1": {
            "complete": False,
            "output_a_minimal_api_array": [],
            "output_b_evidence_packet": {}
        },
        "stage2": {},
        "stage3": {},
        "stage4": {
            "complete": False,
            "veo_scripts": {}
        }
    }

# ========================================
# STAGE RUNNERS
# ========================================

def run_stage1_pipeline(state: dict) -> bool:
    """Run Stage 1: Question Generation"""
    if not ENABLE_STAGE1_QUESTION_GEN:
        log("[STAGE1] ⏭️ SKIPPED (user mode - custom question provided)")
        return False
    
    log_progress("🚀 STARTING STAGE 1: QUESTION GENERATION")
    
    try:
        result = run_stage1(TOPIC, max_questions=MAX_QUESTIONS)
        
        if result is None:
            log("[STAGE1] ❌ FAILED")
            return False
        
        # Update state
        state["stage1"]["complete"] = True
        state["stage1"]["discovery_json"] = result.get("discovery_json", {})
        state["stage1"]["synthesis_json"] = result.get("synthesis_json", {})
        state["stage1"]["output_a_minimal_api_array"] = result.get("output_a_minimal_api_array", [])
        state["stage1"]["output_b_evidence_packet"] = result.get("output_b_evidence_packet", {})
        state["stage"] = "1"
        
        save_state(state, OUTPUT_DIR)
        
        log_progress(f"✅ STAGE 1 COMPLETE: {len(state['stage1']['output_a_minimal_api_array'])} questions generated")
        return True
    
    except Exception as e:
        log(f"[STAGE1] ❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_stage2_for_question(qobj: dict, state: dict) -> bool:
    """Run Stage 2 for a single question: Panshul + Methods"""
    q_id = qobj.get("question_id", "unknown")
    
    log_progress(f"🔬 STARTING STAGE 2 FOR Q {q_id}")
    
    # Set output directory for this question
    set_current_question(q_id)
    
    # Initialize state for this question if needed
    if q_id not in state["stage2"]:
        state["stage2"][q_id] = {
            "panshul": {"complete": False, "result": None},
            "methods": {"complete": False, "results": {}}
        }
    
    try:
        # Run Panshul bot
        panshul_result = None
        if ENABLE_PANSHUL_BOT:
            panshul_result = run_panshul(qobj)
            state["stage2"][q_id]["panshul"]["complete"] = panshul_result is not None
            state["stage2"][q_id]["panshul"]["result"] = panshul_result
        else:
            log("[PANSHUL] ⏭️ SKIPPED (disabled in config)")
        
        # Run method runner (KM/BD/EX)
        method_results = {}
        if ENABLE_METHOD_RUNNER:
            method_results = run_all_methods(qobj)
            state["stage2"][q_id]["methods"]["complete"] = len(method_results) > 0
            state["stage2"][q_id]["methods"]["results"] = method_results
        else:
            log("[METHODS] ⏭️ SKIPPED (disabled in config)")
        
        # Check if we have enough results
        total_methods = (1 if panshul_result else 0) + len(method_results)
        
        if total_methods == 0:
            log(f"[STAGE2] ❌ Q {q_id} - No methods completed")
            return False
        
        log(f"[STAGE2] ✅ Q {q_id} - {total_methods} methods completed")
        
        state["stage"] = "2"
        save_state(state, OUTPUT_DIR)
        
        return True
    
    except Exception as e:
        log(f"[STAGE2] ❌ Q {q_id} EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_stage3_for_question(qobj: dict, state: dict) -> bool:
    """Run Stage 3 for a single question: Combiner"""
    q_id = qobj.get("question_id", "unknown")
    
    log_progress(f"🎯 STARTING STAGE 3 FOR Q {q_id}")
    
    # Set output directory for this question
    set_current_question(q_id)
    
    # Initialize state for this question if needed
    if q_id not in state["stage3"]:
        state["stage3"][q_id] = {
            "complete": False,
            "forecast": None
        }
    
    try:
        # Get results from Stage 2 - wrap Panshul result properly
        panshul_raw = state["stage2"][q_id]["panshul"]["result"]
        panshul_result = panshul_raw  # Already has correct structure
        method_results = state["stage2"][q_id]["methods"]["results"]
        evidence_packet = state["stage1"]["output_b_evidence_packet"]
        
        # Run combiner
        final_forecast = run_stage3(qobj, panshul_result, method_results, evidence_packet)
        
        if final_forecast is None:
            log(f"[STAGE3] ❌ Q {q_id} - Combiner failed")
            return False
        
        # Update state
        state["stage3"][q_id]["complete"] = True
        state["stage3"][q_id]["forecast"] = final_forecast
        state["stage"] = "3"
        
        save_state(state, OUTPUT_DIR)
        
        log_progress(f"✅ STAGE 3 COMPLETE FOR Q {q_id}")
        return True
    
    except Exception as e:
        log(f"[STAGE3] ❌ Q {q_id} EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_stage4_pipeline(state: dict) -> bool:
    """Run Stage 4: Veo Script Generation"""
    if not ENABLE_VEO_GENERATION:
        log("[STAGE4] ⏭️ SKIPPED (disabled in config)")
        return False
    
    log_progress("🎬 STARTING STAGE 4: VEO SCRIPT GENERATION")
    
    try:
        # Collect all forecasts
        forecasts = []
        for q_id, q_state in state["stage3"].items():
            if q_state["complete"] and q_state["forecast"]:
                forecasts.append(q_state["forecast"])
        
        if not forecasts:
            log("[STAGE4] ⚠️ No forecasts to generate scripts from")
            return False
        
        # Generate scripts
        scripts = run_stage4(forecasts)
        
        if scripts is None:
            log("[STAGE4] ❌ FAILED")
            return False
        
        # Update state
        state["stage4"]["complete"] = True
        state["stage4"]["veo_scripts"] = scripts
        state["stage"] = "4"
        
        save_state(state, OUTPUT_DIR)
        
        log_progress(f"✅ STAGE 4 COMPLETE: {len(scripts)} scripts generated")
        return True
    
    except Exception as e:
        log(f"[STAGE4] ❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return False

# ========================================
# MAIN PIPELINE
# ========================================

def run_pipeline():
    """Run the complete forecasting pipeline"""
    log("=" * 80)
    log("🚀 FORECAST PIPELINE - STARTING")
    log("=" * 80)
    
    # Log configuration
    log_configuration()
    
    # Initialize state
    state = initialize_state()
    
    # Try to load existing state (for resume capability)
    existing_state = load_state(OUTPUT_DIR)
    if existing_state:
        log_progress(f"📂 RESUME MODE: Found existing state from stage {existing_state.get('stage', '0')}")
        state = existing_state
    
    # USER MODE: Single question
    if USER_MODE:
        log("=" * 60)
        log("🧑 USER MODE: Running single question")
        log("=" * 60)
        
        if not USER_QUESTION_JSON:
            log("[ERROR] ❌ USER_QUESTION_JSON not set but USER_MODE enabled")
            sys.exit(1)
        
        try:
            qobj = json.loads(USER_QUESTION_JSON)
            q_id = qobj.get("question_id", "user_question")
            log(f"[USER] ✅ Loaded question: Q {q_id}")
            
            # Skip Stage 1 - mark as complete with user's question
            state["stage1"]["complete"] = True
            state["stage1"]["output_a_minimal_api_array"] = [qobj]
            state["stage1"]["output_b_evidence_packet"] = {
                q_id: {
                    "question_text": qobj.get("title", ""),
                    "consequences": {
                        "near_term": "User-provided question",
                        "knock_ons": "User-provided question"
                    }
                }
            }
            
            log(f"[USER] ⏭️ Skipping Stage 1 (question generation)")
            
            # Run Stage 2 & 3 for this question
            if run_stage2_for_question(qobj, state):
                run_stage3_for_question(qobj, state)
            
            # CRITICAL FIX: Run Stage 4 (Veo) in USER mode too
            if ENABLE_VEO_GENERATION:
                log("[USER] Running Stage 4 (Veo generation)")
                run_stage4_pipeline(state)
            
            log("=" * 60)
            log("✅ USER MODE COMPLETE")
            log("=" * 60)
            log(f"[OUTPUT] Results saved to: out/Q_{q_id}/")
            return
        
        except json.JSONDecodeError as e:
            log(f"[ERROR] ❌ Invalid JSON in USER_QUESTION_JSON: {e}")
            sys.exit(1)
        except Exception as e:
            log(f"[ERROR] ❌ User mode failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # NORMAL MODE: Full pipeline
    log("=" * 60)
    log("🤖 NORMAL MODE: Full pipeline")
    log("=" * 60)
    
    # Stage 1: Question Generation
    if not state["stage1"]["complete"]:
        if not run_stage1_pipeline(state):
            log("[FATAL] ❌ Stage 1 failed - aborting")
            sys.exit(1)
    else:
        log("[RESUME] ✅ Stage 1 already complete - skipping")
    
    # Get questions
    questions = state["stage1"]["output_a_minimal_api_array"][:MAX_QUESTIONS]
    log(f"[PIPELINE] 📋 Processing {len(questions)} questions")
    
    # Stages 2 & 3: Loop through each question
    for qobj in questions:
        q_id = qobj.get("question_id")
        
        # Check if already complete
        if q_id in state["stage3"] and state["stage3"][q_id]["complete"]:
            log(f"[RESUME] ✅ Q {q_id} already complete - skipping")
            continue
        
        # Run Stage 2 (methods)
        if run_stage2_for_question(qobj, state):
            # Run Stage 3 (combiner)
            run_stage3_for_question(qobj, state)
    
    # Stage 4: Veo Scripts
    if not state["stage4"]["complete"]:
        run_stage4_pipeline(state)
    else:
        log("[RESUME] ✅ Stage 4 already complete - skipping")
    
    # Final summary
    log("=" * 80)
    log("🎉 PIPELINE COMPLETE")
    log("=" * 80)
    
    completed_questions = sum(1 for q_id in state["stage3"] if state["stage3"][q_id]["complete"])
    log(f"[SUMMARY] Questions completed: {completed_questions}/{len(questions)}")
    log(f"[SUMMARY] Veo scripts: {'✅ Generated' if state['stage4']['complete'] else '❌ Not generated'}")
    log(f"[OUTPUT] All results saved to: {OUTPUT_DIR}/")
    
    # List output directories
    log("\n[OUTPUT DIRECTORIES]")
    for qobj in questions:
        q_id = qobj.get("question_id")
        if q_id in state["stage3"] and state["stage3"][q_id]["complete"]:
            log(f"  ✅ Q {q_id}: out/Q_{q_id}/")
        else:
            log(f"  ❌ Q {q_id}: Failed")

# ========================================
# ENTRY POINT
# ========================================

if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        log("\n[INTERRUPTED] ⚠️ Pipeline stopped by user")
        sys.exit(1)
    except Exception as e:
        log(f"\n[FATAL ERROR] ❌ {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
