#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panshul Bot Wrapper
Provides a clean interface to run the Panshul forecasting bot
"""

import os
import sys
import json
from typing import Dict, Optional
import asyncio

# Add Bot directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Bot'))

from utils import log

# Try to import bot modules
try:
    from Bot.forecaster import binary_forecast, multiple_choice_forecast, numeric_forecast
    from Bot.search import write
    log("[PANSHUL] ✅ Bot modules imported successfully")
    BOT_AVAILABLE = True
except ImportError as e:
    log(f"[PANSHUL] ❌ Failed to import bot modules: {e}")
    BOT_AVAILABLE = False

# ========================================
# CONFIGURATION
# ========================================

# Mode detection
PANSHUL_MODE = os.getenv("PANSHUL_MODE", "full").lower()

if PANSHUL_MODE == "fast":
    log("[PANSHUL] Mode: FAST")
    log("[PANSHUL] Fast mode: ~27 calls, 2 forecasters (Claude + o3), skips Agent/Perplexity")
else:
    log("[PANSHUL] Mode: FULL")
    log("[PANSHUL] Full mode: ~54 calls, 5 forecasters (2 Claude + 1 o4-mini + 2 o3), includes agentic search")

# ========================================
# API AVAILABILITY CHECK
# ========================================

def check_api_availability():
    """Check which APIs are available"""
    HAS_OPENROUTER = bool(os.getenv("OPENROUTER_API_KEY"))
    HAS_SERPER = bool(os.getenv("SERPER_API_KEY") or os.getenv("SERPER_KEY"))
    HAS_ASKNEWS = bool(os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"))
    HAS_NEWSDATA = bool(os.getenv("NEWSDATA_KEY"))
    HAS_NEWSAPI = bool(os.getenv("NEWSAPI_KEY"))
    HAS_PERPLEXITY = bool(os.getenv("PERPLEXITY_API_KEY"))
    HAS_GEMINI = bool(os.getenv("GEMINI_API_KEY"))
    
    log("[PANSHUL] API availability check:")
    log(f"  OpenRouter: {'✅' if HAS_OPENROUTER else '❌'}")
    log(f"  Serper (Google Search): {'✅' if HAS_SERPER else '❌'}")
    
    # News API hierarchy display
    if HAS_ASKNEWS:
        log(f"  AskNews: ✅")
    elif HAS_NEWSDATA:
        log(f"  AskNews: ❌ → Using NewsData.io instead")
    elif HAS_NEWSAPI:
        log(f"  AskNews: ❌ → Using NewsAPI instead")
    else:
        log(f"  AskNews: ❌ → No news APIs available")
    
    if HAS_NEWSDATA:
        log(f"  NewsData.io: ✅")
    
    if HAS_NEWSAPI:
        log(f"  NewsAPI: ✅")
    
    if HAS_PERPLEXITY:
        log(f"  Perplexity: ✅")
    else:
        log(f"  Perplexity: ❌ → Will use agentic search fallback")
    
    log(f"  Gemini: {'✅' if HAS_GEMINI else '❌'}")
    
    return {
        "openrouter": HAS_OPENROUTER,
        "serper": HAS_SERPER,
        "asknews": HAS_ASKNEWS,
        "newsdata": HAS_NEWSDATA,
        "newsapi": HAS_NEWSAPI,
        "perplexity": HAS_PERPLEXITY,
        "gemini": HAS_GEMINI
    }

# ========================================
# QUESTION CONVERSION
# ========================================

def convert_to_panshul_format(qobj: dict) -> dict:
    """
    Convert our question format to Panshul's expected format.
    
    Handles both formats:
    - Metaculus format: question_text, horizon_utc, options
    - Standard format: title, resolution_date, description, resolution_criteria
    """
    # Get question text - try question_text first, then title
    question_text = qobj.get("question_text", qobj.get("title", ""))
    
    # Get description - if not provided, use question_text as fallback
    description = qobj.get("description", "")
    if not description and question_text:
        description = question_text
    
    return {
        "id": qobj.get("question_id", "unknown"),
        "type": qobj.get("question_type", "binary"),
        "title": question_text,
        "resolution_criteria": qobj.get("resolution_criteria", ""),
        "description": description,
        "fine_print": qobj.get("fine_print", ""),
        "resolution_date": qobj.get("horizon_utc", qobj.get("resolution_date", "")),
        "options": qobj.get("options", [])
    }

# ========================================
# MAIN RUNNER
# ========================================

def run_panshul(qobj: dict) -> Optional[Dict]:
    """
    Run Panshul bot on a question.
    
    Args:
        qobj: Question object in our standard format
    
    Returns:
        Dictionary containing forecast results, or None if failed
    """
    log("=" * 60)
    log("🤖 RUNNING PANSHUL BOT")
    log("=" * 60)
    
    # Check if bot is available
    if not BOT_AVAILABLE:
        log("[PANSHUL] ❌ Bot not available (import failed)")
        return None
    
    # Check API availability
    api_status = check_api_availability()
    
    # Convert question format
    panshul_question = convert_to_panshul_format(qobj)
    
    q_id = qobj.get("question_id", "unknown")
    q_type = qobj.get("question_type", "binary")
    
    log(f"[PANSHUL] Running forecast for Q {q_id} (type: {q_type}, mode: {PANSHUL_MODE})")
    
    try:
        # Route to the correct forecast function based on type
        if q_type == "binary":
            result = asyncio.run(binary_forecast(panshul_question))
        elif q_type == "multiple_choice":
            result = asyncio.run(multiple_choice_forecast(panshul_question))
        elif q_type == "numeric":
            result = asyncio.run(numeric_forecast(panshul_question))
        else:
            log(f"[PANSHUL] ❌ Unknown question type: {q_type}")
            return None
        
        if result is None:
            log(f"[PANSHUL] ❌ Bot returned None for Q {q_id}")
            return None
        
        log(f"[PANSHUL] ✅ Forecast complete for Q {q_id}")
        
        # The bot returns (forecast, comment) tuple
        if isinstance(result, tuple) and len(result) == 2:
            forecast, comment = result
            # Return the raw forecast data - combiner will wrap it in PANSHUL_RESULTS
            return {
                "forecast": forecast,
                "comment": comment
            }
        else:
            log(f"[PANSHUL] ⚠️ Unexpected result format: {type(result)}")
            return {
                "forecast": result
            }
    
    except Exception as e:
        log(f"[PANSHUL] ❌ Exception running bot: {e}")
        import traceback
        traceback.print_exc()
        return None

# ========================================
# TESTING
# ========================================

def test_panshul():
    """Test the Panshul wrapper with a sample question"""
    sample_question = {
        "question_id": "test_123",
        "question_type": "binary",
        "title": "Will Bitcoin reach $100,000 by the end of 2025?",
        "resolution_criteria": "This question resolves as YES if Bitcoin (BTC) reaches or exceeds $100,000 USD on any major cryptocurrency exchange by December 31, 2025, 23:59:59 UTC.",
        "description": "Bitcoin has been experiencing significant volatility in 2024-2025. This question asks whether it will reach the psychological milestone of $100,000 before the end of 2025.",
        "fine_print": "The price must be sustained for at least 1 hour on a major exchange (Coinbase, Binance, Kraken, or Bitstamp).",
        "resolution_date": "2025-12-31T23:59:59Z"
    }
    
    result = run_panshul(sample_question)
    
    if result:
        print("\n" + "=" * 60)
        print("PANSHUL BOT RESULT:")
        print("=" * 60)
        print(json.dumps(result, indent=2))
    else:
        print("\n[TEST] ❌ Bot failed to produce result")

if __name__ == "__main__":
    test_panshul()
