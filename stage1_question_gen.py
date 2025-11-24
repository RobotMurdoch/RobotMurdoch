#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast Pipeline - Stage 1: Question Generation
Runs NYT API → RS1 → RS2 → QG chain to generate forecasting questions.
Outputs:
  - OUTPUT_A_MINIMAL_API_ARRAY: List of 3 questions
  - OUTPUT_B_EVIDENCE_PACKET: Evidence/consequences for each question
"""

import os
import json
import re
import aiohttp
import asyncio
from typing import Tuple, Optional, Dict, List
from utils import (
    log, log_progress, save_response, call_llm,
    extract_json_blocks, is_retryable_error,
    MAX_RETRIES, RETRY_DELAYS
)
import time

# Get models and prompts from environment
MODEL_QG = os.getenv("MODEL_QG", "").strip()
PROMPT_RS1 = os.getenv("PROMPT_RS1", "").strip()
PROMPT_RS2 = os.getenv("PROMPT_RS2", "").strip()
PROMPT_QG = os.getenv("PROMPT_QG", "").strip()
NYT_API_KEY = os.getenv("NYT_API_KEY", "").strip()

# Validate
if not MODEL_QG:
    log("[FATAL] MODEL_QG must be set in environment")
    raise ValueError("MODEL_QG not set")

# ========================================
# NYT API: FETCH CURATED ARTICLES
# ========================================

async def fetch_nyt_articles() -> str:
    """
    Fetch top 5 articles each from NYT Top Stories: home, us, world.
    Returns formatted string for inclusion in RS1 prompt.
    """
    if not NYT_API_KEY:
        log("[NYT] ⚠️ NYT_API_KEY not configured - skipping NYT context")
        return ""
    
    log("[NYT] Fetching curated articles from Top Stories API...")
    
    endpoints = [
        ("home", "https://api.nytimes.com/svc/topstories/v2/home.json"),
        ("us", "https://api.nytimes.com/svc/topstories/v2/us.json"),
        ("world", "https://api.nytimes.com/svc/topstories/v2/world.json")
    ]
    
    all_articles = []
    
    async with aiohttp.ClientSession() as session:
        for section_name, url in endpoints:
            try:
                params = {"api-key": NYT_API_KEY}
                timeout = aiohttp.ClientTimeout(total=30)
                
                async with session.get(url, params=params, timeout=timeout) as response:
                    if response.status == 429:
                        log(f"[NYT] ⚠️ Rate limit hit for {section_name} (500/day, 5/min)")
                        continue
                    
                    if response.status != 200:
                        error_text = await response.text()
                        log(f"[NYT] ⚠️ Error {response.status} for {section_name}: {error_text[:200]}")
                        continue
                    
                    data = await response.json()
                    
                    if data.get("status") != "OK":
                        log(f"[NYT] ⚠️ API returned status: {data.get('status')} for {section_name}")
                        continue
                    
                    results = data.get("results", [])
                    
                    # Take top 5 articles from this section
                    for article in results[:5]:
                        title = article.get("title", "No title")
                        abstract = article.get("abstract", "No description")
                        published = article.get("published_date", "Unknown date")
                        url_link = article.get("url", "")
                        
                        all_articles.append({
                            "section": section_name,
                            "title": title,
                            "abstract": abstract,
                            "published": published,
                            "url": url_link
                        })
                    
                    log(f"[NYT] ✅ Fetched {len(results[:5])} articles from {section_name}")
                
                # Small delay between requests to respect rate limits
                await asyncio.sleep(0.2)
                
            except asyncio.TimeoutError:
                log(f"[NYT] ⏱️ Timeout for {section_name}")
                continue
            except Exception as e:
                log(f"[NYT] ❌ Error fetching {section_name}: {str(e)}")
                continue
    
    if not all_articles:
        log("[NYT] ⚠️ No articles fetched from any endpoint")
        return ""
    
    # Format articles for inclusion in prompt
    formatted = "=== NYT CURATED ARTICLES (STARTING CONTEXT) ===\n\n"
    formatted += f"The following {len(all_articles)} articles from NYT Top Stories (home/US/world) are provided as starting context. Use these as leads, but actively search beyond them for the full evidence base.\n\n"
    
    for i, article in enumerate(all_articles, 1):
        formatted += f"{i}. [{article['section'].upper()}] {article['title']}\n"
        formatted += f"   {article['abstract']}\n"
        formatted += f"   Published: {article['published']}\n"
        formatted += f"   URL: {article['url']}\n\n"
    
    formatted += "=== END NYT CONTEXT ===\n\n"
    
    log(f"[NYT] ✅ Formatted {len(all_articles)} articles for RS1")
    
    # Save for reference
    save_response("nyt_context.json", json.dumps(all_articles, indent=2))
    
    return formatted

# ========================================
# RS1: RESEARCH DISCOVERY
# ========================================

def run_rs1(topic: str, max_retries: int = MAX_RETRIES) -> Tuple[str, Optional[dict]]:
    """
    Run RS1 (Research Discovery) with NYT context.
    Returns: (full_response, discovery_json or None)
    """
    if not PROMPT_RS1:
        log("[RS1] ⏭️ SKIPPED (PROMPT_RS1 not configured)")
        return "", None
    
    log_progress("🔍 RUNNING RS1: RESEARCH DISCOVERY")
    
    # Fetch NYT articles first (synchronously wrap async)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    nyt_context = loop.run_until_complete(fetch_nyt_articles())
    loop.close()
    
    # Prepend NYT context to user payload
    user_payload = ""
    if nyt_context:
        user_payload += nyt_context
    user_payload += f"Topic: {topic}"
    
    response = ""
    discovery_json = None
    
    for attempt in range(1, max_retries + 1):
        response = call_llm(
            MODEL_QG,
            PROMPT_RS1,
            user_payload,
            max_tokens=16000,
            timeout=180
        )
        log(f"[RS1] 📥 Received {len(response)} chars (attempt {attempt}/{max_retries})")
        
        # Extract JSON
        blocks = extract_json_blocks(response, "RS1")
        
        for raw in blocks:
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and len(obj) > 0:
                    discovery_json = obj
                    log(f"[RS1] ✅ Valid discovery JSON extracted ({len(raw)} chars)")
                    break
            except Exception:
                continue
        
        if discovery_json:
            break
        
        # Retry if error is retryable
        if is_retryable_error(response) and attempt < max_retries:
            delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            log(f"[RS1] ⚠️ Retryable error - waiting {delay}s before retry {attempt+1}")
            time.sleep(delay)
            continue
    
    if discovery_json is None:
        log(f"[RS1] ❌ FAILED after {max_retries} attempts")
    
    save_response("rs1_discovery_full.txt", response)
    if discovery_json:
        save_response("rs1_discovery.json", json.dumps(discovery_json, indent=2))
    
    return response, discovery_json

# ========================================
# RS2: RESEARCH SYNTHESIS
# ========================================

def run_rs2(discovery_json: dict, max_retries: int = MAX_RETRIES) -> Tuple[str, Optional[dict]]:
    """
    Run RS2 (Research Synthesis).
    Takes discovery_json from RS1, produces synthesis_json.
    Returns: (full_response, synthesis_json or None)
    """
    if not PROMPT_RS2:
        log("[RS2] ⏭️ SKIPPED (PROMPT_RS2 not configured)")
        return "", None
    
    log_progress("🔬 RUNNING RS2: RESEARCH SYNTHESIS")
    
    user_payload = "DISCOVERY_JSON:\n" + json.dumps(discovery_json, indent=2)
    response = ""
    synthesis_json = None
    
    for attempt in range(1, max_retries + 1):
        response = call_llm(
            MODEL_QG,
            PROMPT_RS2,
            user_payload,
            max_tokens=24000,
            timeout=240
        )
        log(f"[RS2] 📥 Received {len(response)} chars (attempt {attempt}/{max_retries})")
        
        # Extract JSON
        blocks = extract_json_blocks(response, "RS2")
        
        for raw in blocks:
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict) and len(obj) > 0:
                    synthesis_json = obj
                    log(f"[RS2] ✅ Valid synthesis JSON extracted ({len(raw)} chars)")
                    break
            except Exception:
                continue
        
        if synthesis_json:
            break
        
        # Retry if error is retryable
        if is_retryable_error(response) and attempt < max_retries:
            delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            log(f"[RS2] ⚠️ Retryable error - waiting {delay}s before retry {attempt+1}")
            time.sleep(delay)
            continue
    
    if synthesis_json is None:
        log(f"[RS2] ❌ FAILED after {max_retries} attempts")
    
    save_response("rs2_synthesis_full.txt", response)
    if synthesis_json:
        save_response("rs2_synthesis.json", json.dumps(synthesis_json, indent=2))
    
    return response, synthesis_json

# ========================================
# QG: QUESTION GENERATOR
# ========================================

def run_qg(
    discovery_json: dict,
    synthesis_json: dict,
    max_questions: int = 3,
    max_retries: int = MAX_RETRIES
) -> Tuple[str, Optional[List[dict]], Optional[dict]]:
    """
    Run QG (Question Generator).
    Takes discovery + synthesis, produces:
      - OUTPUT_A_MINIMAL_API_ARRAY: list of questions
      - OUTPUT_B_EVIDENCE_PACKET: evidence/consequences
    
    Returns: (full_response, output_a or None, output_b or None)
    """
    if not PROMPT_QG:
        log("[QG] ⏭️ SKIPPED (PROMPT_QG not configured)")
        return "", None, None
    
    log_progress("❓ RUNNING QG: QUESTION GENERATION")
    
    user_lines = [
        "DISCOVERY_JSON:",
        json.dumps(discovery_json, indent=2),
        "",
        "SYNTHESIS_JSON:",
        json.dumps(synthesis_json, indent=2),
        "",
        f"Generate exactly {max_questions} forecasting questions."
    ]
    user_payload = "\n".join(user_lines)
    
    response = ""
    output_a = None
    output_b = None
    
    for attempt in range(1, max_retries + 1):
        response = call_llm(
            MODEL_QG,
            PROMPT_QG,
            user_payload,
            max_tokens=24000,
            timeout=240
        )
        log(f"[QG] 📥 Received {len(response)} chars (attempt {attempt}/{max_retries})")
        
        # Try to extract OUTPUT_A using BEGIN/END tags first
        output_a_match = re.search(
            r'BEGIN OUTPUT_A_MINIMAL_API_ARRAY\s*\n(.*?)\n\s*END OUTPUT_A_MINIMAL_API_ARRAY',
            response,
            re.DOTALL
        )
        if output_a_match:
            try:
                output_a = json.loads(output_a_match.group(1).strip())
                if isinstance(output_a, list):
                    output_a = output_a[:max_questions]
                    log(f"[QG] ✅ Found OUTPUT_A (via tags): {len(output_a)} questions")
            except Exception as e:
                log(f"[QG] ⚠️ Failed to parse OUTPUT_A from tags: {e}")
        
        # Try to extract OUTPUT_B using BEGIN/END tags
        output_b_match = re.search(
            r'BEGIN OUTPUT_B_EVIDENCE_PACKET\s*\n(.*?)\n\s*END OUTPUT_B_EVIDENCE_PACKET',
            response,
            re.DOTALL
        )
        if output_b_match:
            try:
                output_b = json.loads(output_b_match.group(1).strip())
                log(f"[QG] ✅ Found OUTPUT_B (via tags): evidence for {len(output_b)} questions")
            except Exception as e:
                log(f"[QG] ⚠️ Failed to parse OUTPUT_B from tags: {e}")
        
        # If tags didn't work, fall back to JSON block extraction
        if not output_a or not output_b:
            blocks = extract_json_blocks(response, "QG")
            
            for raw in blocks:
                try:
                    obj = json.loads(raw)
                    
                    # Check if it's an array (OUTPUT_A)
                    if isinstance(obj, list) and len(obj) > 0 and not output_a:
                        # Validate it looks like questions
                        if all(isinstance(q, dict) and "question_id" in q for q in obj):
                            output_a = obj[:max_questions]
                            log(f"[QG] ✅ Found OUTPUT_A (via blocks): {len(output_a)} questions")
                    
                    # Check if it's an object (OUTPUT_B)
                    elif isinstance(obj, dict) and not output_b:
                        # Look for question IDs as keys
                        keys = list(obj.keys())
                        if keys and any(k.startswith("Q") for k in keys):
                            output_b = obj
                            log(f"[QG] ✅ Found OUTPUT_B (via blocks): evidence for {len(obj)} questions")
                
                except Exception as e:
                    continue
        
        # Success if we have both
        if output_a and output_b:
            break
        
        # Retry if error is retryable
        if is_retryable_error(response) and attempt < max_retries:
            delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            log(f"[QG] ⚠️ Retryable error - waiting {delay}s before retry {attempt+1}")
            time.sleep(delay)
            continue
    
    if output_a is None or output_b is None:
        log(f"[QG] ❌ FAILED after {max_retries} attempts")
        if output_a:
            log(f"[QG]   - OUTPUT_A: ✅ Found")
        else:
            log(f"[QG]   - OUTPUT_A: ❌ Missing")
        if output_b:
            log(f"[QG]   - OUTPUT_B: ✅ Found")
        else:
            log(f"[QG]   - OUTPUT_B: ❌ Missing")
    
    save_response("qg_full.txt", response)
    if output_a:
        save_response("qg_output_a.json", json.dumps(output_a, indent=2))
    if output_b:
        save_response("qg_output_b.json", json.dumps(output_b, indent=2))
    
    return response, output_a, output_b

# ========================================
# STAGE 1 ORCHESTRATOR
# ========================================

def run_stage1(topic: str, max_questions: int = 3) -> Optional[dict]:
    """
    Run complete Stage 1: NYT API → RS1 → RS2 → QG
    
    Returns: {
        "discovery_json": {...},
        "synthesis_json": {...},
        "output_a_minimal_api_array": [...],
        "output_b_evidence_packet": {...}
    } or None if failed
    """
    # Create output directory
    os.makedirs("out", exist_ok=True)
    
    log_progress(f"🚀 STARTING STAGE 1: QUESTION GENERATION")
    log(f"[STAGE1] Topic: {topic}")
    log(f"[STAGE1] Max questions: {max_questions}")
    
    # RS1: Discovery (with NYT context)
    _, discovery_json = run_rs1(topic)
    if discovery_json is None:
        log("[STAGE1] ❌ FAILED at RS1")
        return None
    
    # RS2: Synthesis
    _, synthesis_json = run_rs2(discovery_json)
    if synthesis_json is None:
        log("[STAGE1] ❌ FAILED at RS2")
        return None
    
    # QG: Question Generation
    _, output_a, output_b = run_qg(discovery_json, synthesis_json, max_questions)
    if output_a is None or output_b is None:
        log("[STAGE1] ❌ FAILED at QG")
        return None
    
    # Package results
    result = {
        "discovery_json": discovery_json,
        "synthesis_json": synthesis_json,
        "output_a_minimal_api_array": output_a,
        "output_b_evidence_packet": output_b
    }
    
    # Save complete stage1 output
    save_response("stage1_complete.json", json.dumps(result, indent=2))
    
    log_progress(f"✅ STAGE 1 COMPLETE: Generated {len(output_a)} questions")
    
    return result

# ========================================
# END OF STAGE1_QUESTION_GEN
# ========================================
