#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forecast Pipeline - Stage 4: Veo + ElevenLabs + Sora2 Video Generation

"""

import os
import json
import time
import subprocess
import pickle
import re
from typing import Dict, List, Optional
from pathlib import Path
from google import genai
from google.genai import types
from elevenlabs.client import ElevenLabs
from openai import OpenAI  # ONLY used for Sora2 fallback

from utils import (
    log, log_progress, save_response, call_llm,
    extract_json_blocks, is_retryable_error,
    MAX_RETRIES, RETRY_DELAYS, set_current_question
)

# ========================================
# 🎛️ EASY CONFIGURATION
# ========================================

# ElevenLabs Configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
ELEVENLABS_VOICE_ID = "TK"  # 🔧 EASY TO SWAP!

# Rate Limiting
MAX_VEO_GENERATIONS_PER_RUN = None  # 🔧 EASY TO CHANGE! (set to None for unlimited)

# Sora2 Fallback Configuration
ENABLE_SORA2_FALLBACK = True  # 🔧 Toggle Sora2 fallback on/off
SORA2_MODEL = "sora-2"  # 🔧 Options: "sora-2" (fast) or "sora-2-pro" (quality)

# 🔧 SORA2 TYPE 1 ANCHOR DESCRIPTION (ONLY for Type 1 Sora2 fallback - easy to edit!)
SORA2_ANCHOR_DESCRIPTION = """A professional news anchor at a broadcast desk. Medium close-up shot with anchor positioned on left third of frame, leaving empty space over right shoulder for graphics. Direct eye contact with camera, serious authoritative expression. Business professional attire. Modern news studio with soft lighting, blue and neutral tones in background. Static composition, no camera movement. No text overlays, no chyrons, no graphics, clean frame. High quality broadcast photography, photorealistic."""

# Retry Configuration
ELEVENLABS_MAX_RETRIES = 3
ELEVENLABS_RETRY_DELAYS = [5, 10, 20]  # seconds

# Model Configuration - CHANGED to Claude 3.5 Sonnet for REPORTER prompts
PROMPT_REPORTER1 = os.getenv("PROMPT_REPORTER1", "").strip()
PROMPT_REPORTER2 = os.getenv("PROMPT_REPORTER2", "").strip()
MODEL_REPORTER1 = "anthropic/claude-sonnet-4.5"
MODEL_REPORTER2 = "anthropic/claude-3.5-sonnet"

# Gemini/Veo Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
VEO_MODEL = "veo-3.1-generate-preview"
STOP_AFTER_VEO_SCRIPTS = os.getenv("STOP_AFTER_VEO_SCRIPTS", "false").lower() == "true"

# OpenAI Configuration (ONLY for Sora2 fallback - not used for any LLM calls)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Initialize clients
gemini_client = None
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

elevenlabs_client = None
if ELEVENLABS_API_KEY:
    elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# OpenAI client - ONLY for Sora2 fallback (not used for LLM calls!)
openai_client = None
if OPENAI_API_KEY and ENABLE_SORA2_FALLBACK:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ========================================
# 🔄 ELEVENLABS SMART RETRY WRAPPER
# ========================================

def call_elevenlabs_with_retry(operation_name: str, func, *args, **kwargs):
    """
    Smart retry wrapper for ElevenLabs API calls.
    Handles: rate limits, network errors, service errors
    """
    for attempt in range(1, ELEVENLABS_MAX_RETRIES + 1):
        try:
            log(f"[ELEVENLABS] 🎙️ {operation_name} - Attempt {attempt}/{ELEVENLABS_MAX_RETRIES}")
            result = func(*args, **kwargs)
            log(f"[ELEVENLABS] ✅ {operation_name} succeeded")
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if retryable
            is_retryable = any([
                'rate limit' in error_msg,
                'quota' in error_msg,
                'timeout' in error_msg,
                '500' in error_msg,
                '502' in error_msg,
                '503' in error_msg,
                'connection' in error_msg,
                'temporarily unavailable' in error_msg
            ])
            
            if is_retryable and attempt < ELEVENLABS_MAX_RETRIES:
                delay = ELEVENLABS_RETRY_DELAYS[attempt - 1]
                log(f"[ELEVENLABS] ⚠️ {operation_name} attempt {attempt} failed: {e}")
                log(f"[ELEVENLABS] ⏳ Retrying in {delay}s...")
                time.sleep(delay)
                continue
            else:
                log(f"[ELEVENLABS] ❌ {operation_name} FAILED after {attempt} attempts: {e}")
                if not is_retryable:
                    log(f"[ELEVENLABS] 🚫 Non-retryable error (auth/invalid params)")
                raise
    
    return None

# ========================================
# STEP 1: GENERATE SCRIPT (REPORTER1)
# ========================================

def generate_script(forecast: dict) -> Optional[tuple]:
    """
    Use PROMPT_REPORTER1 to convert forecast into 5-sentence news script.
    
    Returns: (script_json, dialogue_only_json) or None if failed
    """
    if not PROMPT_REPORTER1:
        log("[REPORTER1] ⏭️ SKIPPED (PROMPT_REPORTER1 not configured)")
        return None
    
    q_id = forecast.get("metadata", {}).get("question_id", "unknown")
    log(f"[REPORTER1] 📝 Generating script for Q {q_id}")
    
    user_payload = json.dumps(forecast, indent=2)
    
    script = None
    dialogue_only = None
    response = ""
    
    for attempt in range(1, MAX_RETRIES + 1):
        response = call_llm(
            MODEL_REPORTER1,
            PROMPT_REPORTER1,
            user_payload,
            max_tokens=8000,
            timeout=120
        )
        log(f"[REPORTER1] 📥 Received {len(response)} chars (attempt {attempt}/{MAX_RETRIES})")
        
        blocks = extract_json_blocks(response, "REPORTER1")
        
        if len(blocks) >= 2:
            # Should have 2 JSON blocks: script + dialogue_only
            try:
                script = json.loads(blocks[0])
                dialogue_only = json.loads(blocks[1])
                log(f"[REPORTER1] ✅ Valid script + dialogue_only extracted")
                break
            except Exception as e:
                log(f"[REPORTER1] ⚠️ JSON parse error: {e}")
        elif len(blocks) == 1:
            log(f"[REPORTER1] ⚠️ Only got 1 JSON block, expected 2")
        
        if is_retryable_error(response) and attempt < MAX_RETRIES:
            delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            log(f"[REPORTER1] ⚠️ Retrying in {delay}s...")
            time.sleep(delay)
            continue
    
    if script is None or dialogue_only is None:
        log(f"[REPORTER1] ❌ FAILED after {MAX_RETRIES} attempts")
        output_path = os.path.join("out", f"reporter1_q{q_id}_full.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response)
        return None
    
    # Save both JSONs
    script_path = os.path.join("out", f"script_q{q_id}.json")
    with open(script_path, 'w', encoding='utf-8') as f:
        json.dump(script, f, indent=2)
    
    dialogue_path = os.path.join("out", f"dialogue_only_q{q_id}.json")
    with open(dialogue_path, 'w', encoding='utf-8') as f:
        json.dump(dialogue_only, f, indent=2)
    
    log(f"[REPORTER1] 💾 Saved script to {script_path}")
    log(f"[REPORTER1] 💾 Saved dialogue_only to {dialogue_path}")
    
    return (script, dialogue_only)

# ========================================
# STEP 2: GENERATE VEO + ELEVENLABS PROMPTS (REPORTER2)
# ========================================

def generate_veo_elevenlabs_prompts(script: dict) -> Optional[dict]:
    """
    Use PROMPT_REPORTER2 to convert script into Veo + ElevenLabs prompts.
    
    Returns: Combined JSON with veo_prompt, elevenlabs_dialogue, processing_type, etc.
    """
    if not PROMPT_REPORTER2:
        log("[REPORTER2] ⏭️ SKIPPED (PROMPT_REPORTER2 not configured)")
        return None
    
    q_id = script.get("question_id", "unknown")
    log(f"[REPORTER2] 🎬 Generating Veo + ElevenLabs prompts for Q {q_id}")
    
    user_payload = json.dumps(script, indent=2)
    
    combined_prompts = None
    response = ""
    
    for attempt in range(1, MAX_RETRIES + 1):
        response = call_llm(
            MODEL_REPORTER2,
            PROMPT_REPORTER2,
            user_payload,
            max_tokens=16000,
            timeout=180
        )
        log(f"[REPORTER2] 📥 Received {len(response)} chars (attempt {attempt}/{MAX_RETRIES})")
        
        blocks = extract_json_blocks(response, "REPORTER2")
        
        if blocks:
            block = max(blocks, key=len)
            try:
                combined_prompts = json.loads(block)
                log(f"[REPORTER2] ✅ Valid combined prompts extracted")
                break
            except Exception as e:
                log(f"[REPORTER2] ⚠️ JSON parse error: {e}")
        
        if is_retryable_error(response) and attempt < MAX_RETRIES:
            delay = RETRY_DELAYS[min(attempt - 1, len(RETRY_DELAYS) - 1)]
            log(f"[REPORTER2] ⚠️ Retrying in {delay}s...")
            time.sleep(delay)
            continue
    
    if combined_prompts is None:
        log(f"[REPORTER2] ❌ FAILED after {MAX_RETRIES} attempts")
        output_path = os.path.join("out", f"reporter2_q{q_id}_full.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response)
        return None
    
    # Save combined prompts
    output_path = os.path.join("out", f"veo_elevenlabs_prompts_q{q_id}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_prompts, f, indent=2)
    
    log(f"[REPORTER2] 💾 Saved to {output_path}")
    
    return combined_prompts

# ========================================
# STEP 3: LOAD ANCHOR IMAGE
# ========================================

def load_anchor_image():
    """Load the pickled anchor Part object for image-to-video generation."""
    pickle_path = "anchor_part.pkl"
    
    if not os.path.exists(pickle_path):
        log(f"[VEO] ⚠️ Anchor pickle not found: {pickle_path}")
        return None
    
    try:
        log(f"[VEO] 📸 Loading pickled anchor Part: {pickle_path}")
        
        with open(pickle_path, "rb") as f:
            part = pickle.load(f)
        
        anchor_image = part.as_image()
        
        log(f"[VEO] ✅ Loaded anchor image from pickled Part")
        return anchor_image
    
    except Exception as e:
        log(f"[VEO] ❌ Failed to load pickled anchor: {e}")
        import traceback
        traceback.print_exc()
        return None

# ========================================
# STEP 4: ADAPT PROMPT FOR SORA2
# ========================================

def adapt_prompt_for_sora2(veo_prompt: str, processing_type: int, dialogue: str = "") -> str:
    """
    Adapt a Veo prompt for Sora2 by removing Veo-specific elements.
    
    For Type 1 (anchor shots), includes dialogue in the prompt and uses SORA2_ANCHOR_DESCRIPTION.
    """
    prompt = veo_prompt
    
    # Remove Veo-specific flags
    prompt = re.sub(r'--no\s+[^\n]+', '', prompt)
    
    # Remove CONTINUITY LOCK (Veo-specific)
    prompt = re.sub(r'CONTINUITY LOCK:[^\n]+\n?', '', prompt)
    
    # For Type 1 (anchor), replace with Sora2 anchor description and include dialogue
    if processing_type == 1:
        # Remove old ACTION section
        prompt = re.sub(r'ACTION:[^\.]+\.', '', prompt)
        
        # Add Sora2 anchor description + dialogue
        anchor_with_dialogue = f"{SORA2_ANCHOR_DESCRIPTION} The anchor says: \"{dialogue}\""
        prompt = f"STYLE: News broadcast, cinematic, 4K, high fidelity. {anchor_with_dialogue}"
    
    # Clean up extra whitespace
    prompt = re.sub(r'\n\s*\n', '\n', prompt).strip()
    
    return prompt

# ========================================
# STEP 5: GENERATE VEO VIDEO WITH RETRIES (FIXED VERSION FROM OLD CODE)
# ========================================

def generate_veo_video(shot_data: dict, anchor_image, q_id: str, veo_count: dict) -> Optional[str]:
    """
    Generate a single Veo video with retry logic and alternate prompts.
    FIXED: Restored working polling logic from original version
    """
    # Check rate limit
    if MAX_VEO_GENERATIONS_PER_RUN and veo_count["count"] >= MAX_VEO_GENERATIONS_PER_RUN:
        log(f"[VEO] ⏸️ RATE LIMIT REACHED: {veo_count['count']}/{MAX_VEO_GENERATIONS_PER_RUN} videos generated")
        log(f"[VEO] 💡 Run pipeline again to generate more videos (will resume from here)")
        return None
    
    shot_num = shot_data.get("shot", 0)
    use_anchor = shot_data.get("use_reference_image", False)
    veo_prompt = shot_data.get("veo_prompt", "")
    veo_prompt_alts = shot_data.get("veo_prompt_alts", [])
    
    if not veo_prompt:
        log(f"[VEO] ⚠️ Shot {shot_num} has no veo_prompt - skipping")
        return None
    
    if not gemini_client:
        log(f"[VEO] ❌ Gemini client not initialized")
        return None
    
    log(f"[VEO] 🎬 Generating shot {shot_num} for Q {q_id}")
    
    # Try primary prompt + all alternates
    all_prompts = [veo_prompt] + veo_prompt_alts
    
    for idx, prompt_text in enumerate(all_prompts):
        version = "PRIMARY" if idx == 0 else f"ALTERNATE_{idx}"
        
        if not prompt_text:
            continue
        
        log(f"[VEO] 🎯 Trying shot {shot_num} - version '{version}'")
        
        # Try this version up to 2 times
        for attempt in range(1, 3):
            try:
                log(f"[VEO] 📤 Submitting generation request for shot {shot_num}...")
                
                # Use image-to-video for anchor shots (Type 1)
                if use_anchor and anchor_image:
                    log(f"[VEO] 📸 Using image-to-video with anchor image for shot {shot_num}")
                    operation = gemini_client.models.generate_videos(
                        model=VEO_MODEL,
                        prompt=prompt_text,
                        image=anchor_image,
                    )
                else:
                    # Text-to-video only (Type 2 & 3)
                    operation = gemini_client.models.generate_videos(
                        model=VEO_MODEL,
                        prompt=prompt_text,
                    )
                
                # Poll for completion (RESTORED FROM ORIGINAL WORKING VERSION)
                log(f"[VEO] ⏳ Waiting for shot {shot_num} to generate (2-5 minutes)...")
                
                max_wait_time = 600  # 10 minutes max
                elapsed = 0
                poll_interval = 10  # Check every 10 seconds
                
                while not operation.done:
                    if elapsed >= max_wait_time:
                        log(f"[VEO] ⏱️ Shot {shot_num} timed out after {max_wait_time}s")
                        break
                    
                    time.sleep(poll_interval)
                    elapsed += poll_interval
                    
                    if elapsed % 60 == 0:  # Log every minute
                        log(f"[VEO] ⏳ Still waiting... ({elapsed}s elapsed)")
                    
                    # Refresh operation status
                    try:
                        operation = gemini_client.operations.get(operation)
                    except Exception as refresh_error:
                        log(f"[VEO] ⚠️ Error refreshing operation: {refresh_error}")
                        break
                
                if not operation.done:
                    log(f"[VEO] ⏱️ Shot {shot_num} timed out")
                    if attempt < 2:
                        time.sleep(5)
                    continue
                
                # Check if operation succeeded
                if not operation.response:
                    log(f"[VEO] ❌ Shot {shot_num} - No response from Veo")
                    if attempt < 2:
                        time.sleep(5)
                    continue
                
                if not hasattr(operation.response, 'generated_videos') or not operation.response.generated_videos:
                    log(f"[VEO] ❌ Shot {shot_num} - Veo rejected content (safety filters)")
                    log(f"[VEO] 💡 Content blocked - trying next alternate")
                    break  # Don't retry same prompt, go to next alternate
                
                # Download the generated video
                generated_video = operation.response.generated_videos[0]
                
                output_filename = f"Q{q_id}_{shot_num}_VEO.mp4"
                output_path = os.path.join("out", output_filename)
                
                # Download and save
                gemini_client.files.download(file=generated_video.video)
                generated_video.video.save(output_path)
                
                # Increment counter
                veo_count["count"] += 1
                
                log(f"[VEO] ✅ Shot {shot_num} saved to {output_path}")
                log(f"[VEO] 📊 Progress: {veo_count['count']}/{MAX_VEO_GENERATIONS_PER_RUN or 'unlimited'} Veo videos generated")
                
                return output_path
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if it's a rate limit (skip alternates)
                if any(x in error_msg for x in ['rate limit', 'quota', '429', 'too many requests']):
                    log(f"[VEO] ⏸️ Shot {shot_num} - Rate limit detected")
                    log(f"[VEO] 💡 Skipping remaining alternates - will try Sora2 fallback")
                    return None
                
                log(f"[VEO] ⚠️ Shot {shot_num} attempt {attempt}/2 failed: {e}")
                import traceback
                traceback.print_exc()
                
                if attempt < 2:
                    time.sleep(5)
                    continue
        
        log(f"[VEO] ⚠️ Shot {shot_num} version '{version}' failed - trying next version")
    
    log(f"[VEO] ❌ Shot {shot_num} - all Veo versions failed")
    return None

# ========================================
# STEP 6: GENERATE SORA2 VIDEO (FALLBACK)
# ========================================

def generate_sora2_video(shot_data: dict, q_id: str, veo_count: dict) -> Optional[str]:
    """
    Generate a single Sora2 video as fallback when Veo fails.
    Uses the same prompts (adapted for Sora2).
    
    Returns: Path to generated SORA MP4 file, or None if failed
    """
    if not ENABLE_SORA2_FALLBACK:
        log(f"[SORA2] ⏭️ Fallback disabled in config")
        return None
    
    if not openai_client:
        log(f"[SORA2] ❌ OpenAI client not initialized")
        return None
    
    shot_num = shot_data.get("shot", 0)
    processing_type = shot_data.get("processing_type", 2)
    veo_prompt = shot_data.get("veo_prompt", "")
    veo_prompt_alts = shot_data.get("veo_prompt_alts", [])
    dialogue = shot_data.get("elevenlabs_dialogue", "")
    
    log(f"[SORA2] 🎬 Attempting Sora2 fallback for shot {shot_num}")
    
    # Adapt all prompts for Sora2
    all_prompts = [veo_prompt] + veo_prompt_alts
    adapted_prompts = [adapt_prompt_for_sora2(p, processing_type, dialogue) for p in all_prompts if p]
    
    for idx, prompt_text in enumerate(adapted_prompts):
        version = "PRIMARY" if idx == 0 else f"ALTERNATE_{idx}"
        
        log(f"[SORA2] 🎯 Trying shot {shot_num} - version '{version}'")
        
        try:
            log(f"[SORA2] 📤 Submitting generation request...")
            
            # Create video job
            video_job = openai_client.videos.create(
                model=SORA2_MODEL,
                prompt=prompt_text,
                size="1280x720",
                seconds="8"
            )
            
            log(f"[SORA2] ⏳ Waiting for shot {shot_num} to generate...")
            
            # Poll for completion
            max_wait_time = 600  # 10 minutes
            elapsed = 0
            poll_interval = 10
            
            while video_job.status in ["queued", "in_progress"]:
                if elapsed >= max_wait_time:
                    log(f"[SORA2] ⏱️ Shot {shot_num} timed out")
                    break
                
                time.sleep(poll_interval)
                elapsed += poll_interval
                
                if elapsed % 60 == 0:
                    log(f"[SORA2] ⏳ Still waiting... ({elapsed}s elapsed)")
                
                video_job = openai_client.videos.retrieve(video_job.id)
            
            if video_job.status != "completed":
                log(f"[SORA2] ❌ Shot {shot_num} failed with status: {video_job.status}")
                continue
            
            # Download the video
            log(f"[SORA2] 📥 Downloading video...")
            
            video_content = openai_client.videos.download_content(video_job.id)
            
            output_filename = f"Q{q_id}_{shot_num}_SORA.mp4"
            output_path = os.path.join("out", output_filename)
            
            # Save video
            video_data = video_content.read()
            with open(output_path, 'wb') as f:
                f.write(video_data)
            
            # Increment counter (share with Veo count)
            veo_count["count"] += 1
            
            log(f"[SORA2] ✅ Shot {shot_num} saved to {output_path}")
            log(f"[SORA2] 📊 Progress: {veo_count['count']}/{MAX_VEO_GENERATIONS_PER_RUN or 'unlimited'} videos generated")
            
            return output_path
            
        except Exception as e:
            log(f"[SORA2] ⚠️ Shot {shot_num} version '{version}' failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    log(f"[SORA2] ❌ Shot {shot_num} - all Sora2 versions failed")
    return None

# ========================================
# STEP 7: ELEVENLABS TEXT-TO-SPEECH
# ========================================

def generate_elevenlabs_audio(dialogue: str, shot_num: int, q_id: str) -> Optional[str]:
    """
    Generate audio using ElevenLabs TTS.
    
    Returns: Path to generated MP3 file, or None if failed
    """
    if not elevenlabs_client:
        log(f"[ELEVENLABS] ❌ Client not initialized")
        return None
    
    if not dialogue or not dialogue.strip():
        log(f"[ELEVENLABS] ⚠️ Shot {shot_num} - no dialogue provided")
        return None
    
    log(f"[ELEVENLABS] 🎙️ Generating TTS for shot {shot_num} Q {q_id}")
    log(f"[ELEVENLABS] 📝 Dialogue: \"{dialogue[:100]}...\"" if len(dialogue) > 100 else f"[ELEVENLABS] 📝 Dialogue: \"{dialogue}\"")
    
    def _generate():
        audio_generator = elevenlabs_client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=dialogue,
            model_id="eleven_multilingual_v2",
            voice_settings={
                "stability": 0.4,           # Lower = more expressive (0.3-0.5 for news)
                "similarity_boost": 0.75,    # Keep voice consistent
                "style": 0.3,                # Add natural emotion/inflection
                "use_speaker_boost": True    # Enhance clarity
            }
        )
        
        # Collect all audio chunks
        audio_data = b"".join(audio_generator)
        return audio_data
    
    try:
        audio_data = call_elevenlabs_with_retry(
            f"TTS for shot {shot_num}",
            _generate
        )
        
        if audio_data:
            output_filename = f"Q{q_id}_{shot_num}_AUDIO.mp3"
            output_path = os.path.join("out", output_filename)
            
            with open(output_path, 'wb') as f:
                f.write(audio_data)
            
            log(f"[ELEVENLABS] ✅ TTS saved to {output_path}")
            return output_path
        else:
            return None
            
    except Exception as e:
        log(f"[ELEVENLABS] ❌ TTS failed for shot {shot_num}: {e}")
        return None

# ========================================
# STEP 8: ELEVENLABS VOICE CHANGER (FIXED)
# ========================================

def convert_voice_with_elevenlabs(video_path: str, shot_num: int, q_id: str) -> Optional[str]:
    """
    Convert voice in video using ElevenLabs Speech-to-Speech (Voice Changer).
    
    Args:
        video_path: Path to Veo/Sora-generated video with dialogue
        shot_num: Shot number
        q_id: Question ID
    
    Returns: Path to voice-converted video, or None if failed
    """
    if not elevenlabs_client:
        log(f"[ELEVENLABS] ❌ Client not initialized")
        return None
    
    if not os.path.exists(video_path):
        log(f"[ELEVENLABS] ❌ Video file not found: {video_path}")
        return None
    
    log(f"[ELEVENLABS] 🎙️ Converting voice for shot {shot_num} Q {q_id}")
    log(f"[ELEVENLABS] 📹 Input video: {video_path}")
    
    def _convert():
        with open(video_path, 'rb') as video_file:
            audio_generator = elevenlabs_client.speech_to_speech.convert(
                voice_id=ELEVENLABS_VOICE_ID,
                audio=video_file,
                model_id="eleven_multilingual_sts_v2",
                voice_settings=json.dumps({  # ✅ FIXED: json.dumps() for speech-to-speech API
                    "stability": 0.4,
                    "similarity_boost": 0.75,
                    "style": 0.3,
                    "use_speaker_boost": True
                })
            )
            
            # Collect all audio chunks
            audio_data = b"".join(audio_generator)
            return audio_data
    
    try:
        converted_audio = call_elevenlabs_with_retry(
            f"Voice Changer for shot {shot_num}",
            _convert
        )
        
        if not converted_audio:
            return None
        
        # Save converted audio
        audio_filename = f"Q{q_id}_{shot_num}_CONVERTED_AUDIO.mp3"
        audio_path = os.path.join("out", audio_filename)
        
        with open(audio_path, 'wb') as f:
            f.write(converted_audio)
        
        log(f"[ELEVENLABS] ✅ Converted audio saved to {audio_path}")
        
        # Now merge converted audio back with original video
        output_filename = f"Q{q_id}_{shot_num}_PROCESSED.mp4"
        output_path = os.path.join("out", output_filename)
        
        log(f"[FFMPEG] 🎬 Merging converted audio with video...")
        
        # Use just filenames since we're running from 'out' directory
        video_filename = os.path.basename(video_path)
        audio_filename_base = os.path.basename(audio_path)
        output_filename_base = os.path.basename(output_path)
        
        result = subprocess.run([
            'ffmpeg',
            '-i', video_filename,      # Just filename
            '-i', audio_filename_base, # Just filename
            '-c:v', 'copy',            # Copy video stream
            '-map', '0:v:0',           # Use video from first input
            '-map', '1:a:0',           # Use audio from second input
            '-shortest',               # Match shortest stream
            '-y',                      # Overwrite output
            output_filename_base       # Just filename
        ], capture_output=True, text=True, cwd='out')
        
        if result.returncode != 0:
            log(f"[FFMPEG] ❌ Failed to merge: {result.stderr}")
            return None
        
        log(f"[FFMPEG] ✅ Voice-converted video saved to {output_path}")
        
        # Clean up intermediate audio (but keep original video)
        try:
            os.remove(audio_path)
            log(f"[CLEANUP] 🗑️ Removed intermediate audio: {audio_path}")
        except:
            pass
        
        # SKIP: Don't delete original VEO/SORA video 
        
        return output_path
        
    except Exception as e:
        log(f"[ELEVENLABS] ❌ Voice conversion failed for shot {shot_num}: {e}")
        import traceback
        traceback.print_exc()
        return None

# ========================================
# STEP 9: MERGE VEO/SORA VIDEO + ELEVENLABS AUDIO
# ========================================

def merge_video_and_audio(video_path: str, audio_path: str, shot_num: int, q_id: str) -> Optional[str]:
    """
    Merge Veo/Sora-generated video (visuals only) with ElevenLabs-generated audio.
    
    Returns: Path to merged video (Q#_#_PROCESSED.mp4), or None if failed
    """
    if not os.path.exists(video_path):
        log(f"[FFMPEG] ❌ Video file not found: {video_path}")
        return None
    
    if not os.path.exists(audio_path):
        log(f"[FFMPEG] ❌ Audio file not found: {audio_path}")
        return None
    
    log(f"[FFMPEG] 🎬 Merging video + audio for shot {shot_num} Q {q_id}")
    log(f"[FFMPEG] 📹 Video: {video_path}")
    log(f"[FFMPEG] 🎙️ Audio: {audio_path}")
    
    output_filename = f"Q{q_id}_{shot_num}_PROCESSED.mp4"
    output_path = os.path.join("out", output_filename)
    
    try:
        # Use just filenames since we're running from 'out' directory
        video_filename = os.path.basename(video_path)
        audio_filename = os.path.basename(audio_path)
        output_filename_base = os.path.basename(output_path)
        
        result = subprocess.run([
            'ffmpeg',
            '-i', video_filename,      # Just filename
            '-i', audio_filename,      # Just filename
            '-c:v', 'copy',            # Copy video stream (no re-encoding)
            '-map', '0:v:0',           # Use video from first input
            '-map', '1:a:0',           # Use audio from second input
            '-shortest',               # Match shortest stream duration
            '-y',                      # Overwrite output
            output_filename_base       # Just filename
        ], capture_output=True, text=True, cwd='out')
        
        if result.returncode != 0:
            log(f"[FFMPEG] ❌ Merge failed: {result.stderr}")
            return None
        
        log(f"[FFMPEG] ✅ Merged video saved to {output_path}")
        
        # Clean up intermediate audio (but keep original video)
        try:
            os.remove(audio_path)
            log(f"[CLEANUP] 🗑️ Removed intermediate audio: {audio_path}")
        except:
            pass
        
        # SKIP: Don't delete original VEO/SORA video 
        
        return output_path
        
    except Exception as e:
        log(f"[FFMPEG] ❌ Merge failed for shot {shot_num}: {e}")
        import traceback
        traceback.print_exc()
        return None

# ========================================
# STEP 10: GENERATE ADR AUDIO
# ========================================

def generate_adr_audio(dialogue_only_json: dict, q_id: str) -> Optional[str]:
    """
    Generate ADR (backup voiceover) audio from dialogue_only JSON.
    
    Returns: Path to ADR.mp3 file, or None if failed
    """
    if not elevenlabs_client:
        log(f"[ADR] ❌ ElevenLabs client not initialized")
        return None
    
    full_dialogue = dialogue_only_json.get("full_dialogue", "")
    
    if not full_dialogue or not full_dialogue.strip():
        log(f"[ADR] ⚠️ No dialogue found in dialogue_only JSON for Q {q_id}")
        return None
    
    log(f"[ADR] 🎙️ Generating ADR audio for Q {q_id}")
    log(f"[ADR] 📝 Full dialogue ({len(full_dialogue)} chars)")
    
    def _generate():
        audio_generator = elevenlabs_client.text_to_speech.convert(
            voice_id=ELEVENLABS_VOICE_ID,
            text=full_dialogue,
            model_id="eleven_multilingual_v2",
            voice_settings={
                "stability": 0.4,
                "similarity_boost": 0.75,
                "style": 0.3,
                "use_speaker_boost": True
            }
        )
        
        audio_data = b"".join(audio_generator)
        return audio_data
    
    try:
        audio_data = call_elevenlabs_with_retry(
            f"ADR for Q {q_id}",
            _generate
        )
        
        if audio_data:
            output_filename = f"ADR_Q{q_id}.mp3"
            output_path = os.path.join("out", output_filename)
            
            with open(output_path, 'wb') as f:
                f.write(audio_data)
            
            log(f"[ADR] ✅ ADR audio saved to {output_path}")
            return output_path
        else:
            return None
            
    except Exception as e:
        log(f"[ADR] ❌ ADR generation failed for Q {q_id}: {e}")
        return None

# ========================================
# STEP 11: PROCESS SINGLE SHOT (FIXED SORA2 FALLBACK)
# ========================================

def process_shot(shot_data: dict, anchor_image, q_id: str, veo_count: dict) -> Optional[str]:
    """
    Process a single shot based on processing_type.
    Tries Veo first (all alternates), then Sora2 as fallback if under rate limit.
    
    Returns: Path to final *_PROCESSED.mp4 file, or None if failed/rate limited
    """
    shot_num = shot_data.get("shot", 0)
    processing_type = shot_data.get("processing_type", 2)
    
    log(f"\n{'='*60}")
    log(f"[SHOT {shot_num}] 🎬 PROCESSING TYPE {processing_type}")
    log(f"{'='*60}")
    
    # Step 1: Try Veo first (all alternates)
    video_path = generate_veo_video(shot_data, anchor_image, q_id, veo_count)
    
    # Step 2: If Veo failed, decide whether to try Sora2
    if not video_path:
        # Check if we already hit rate limit during Veo attempts
        if MAX_VEO_GENERATIONS_PER_RUN and veo_count["count"] >= MAX_VEO_GENERATIONS_PER_RUN:
            log(f"[SHOT {shot_num}] ⏸️ RATE LIMIT REACHED ({veo_count['count']}/{MAX_VEO_GENERATIONS_PER_RUN})")
            log(f"[SHOT {shot_num}] 💡 Run pipeline again to continue generating videos")
            return None  # ✅ FIXED: Stop here - don't try Sora2 if at limit
        
        # Veo failed for content/other reasons (not rate limit) - try Sora2
        log(f"[SHOT {shot_num}] 🔄 All Veo attempts failed - trying Sora2 fallback")
        video_path = generate_sora2_video(shot_data, q_id, veo_count)
        
        if not video_path:
            log(f"[SHOT {shot_num}] ❌ Both Veo and Sora2 failed")
            return None
    
    # Now process based on type
    if processing_type == 1:
        # TYPE 1: Anchor with Voice Changer
        log(f"[SHOT {shot_num}] 🎙️ Type 1: Applying Voice Changer")
        processed_path = convert_voice_with_elevenlabs(video_path, shot_num, q_id)
        
        if not processed_path:
            log(f"[SHOT {shot_num}] ⚠️ Voice Changer failed - keeping original video")
            # Fallback: copy video to PROCESSED
            fallback_path = os.path.join("out", f"Q{q_id}_{shot_num}_PROCESSED.mp4")
            
            import shutil
            shutil.copy2(video_path, fallback_path)
            
            return fallback_path
        
        return processed_path
    
    elif processing_type == 2:
        # TYPE 2: Video visuals + ElevenLabs TTS voiceover
        log(f"[SHOT {shot_num}] 🎙️ Type 2: Generating TTS voiceover")
        
        dialogue = shot_data.get("elevenlabs_dialogue", "")
        
        if not dialogue:
            log(f"[SHOT {shot_num}] ⚠️ No dialogue for Type 2 - using video as-is")
            processed_path = os.path.join("out", f"Q{q_id}_{shot_num}_PROCESSED.mp4")
            
            import shutil
            shutil.copy2(video_path, processed_path)
            
            return processed_path
        
        # Generate TTS audio
        audio_path = generate_elevenlabs_audio(dialogue, shot_num, q_id)
        
        if not audio_path:
            log(f"[SHOT {shot_num}] ⚠️ TTS failed - using video as-is")
            processed_path = os.path.join("out", f"Q{q_id}_{shot_num}_PROCESSED.mp4")
            
            import shutil
            shutil.copy2(video_path, processed_path)
            
            return processed_path
        
        # Merge video + audio
        processed_path = merge_video_and_audio(video_path, audio_path, shot_num, q_id)
        
        if not processed_path:
            log(f"[SHOT {shot_num}] ⚠️ Merge failed - using video as-is")
            processed_path = os.path.join("out", f"Q{q_id}_{shot_num}_PROCESSED.mp4")
            
            import shutil
            shutil.copy2(video_path, processed_path)
            
            return processed_path
        
        return processed_path
    
    elif processing_type == 3:
        # TYPE 3: Eyewitness - leave as-is, just copy to PROCESSED
        log(f"[SHOT {shot_num}] 👁️ Type 3: Eyewitness - leaving as-is")
        
        processed_path = os.path.join("out", f"Q{q_id}_{shot_num}_PROCESSED.mp4")
        
        import shutil
        shutil.copy2(video_path, processed_path)
        
        log(f"[SHOT {shot_num}] ✅ Copied to {processed_path}")
        return processed_path
    
    else:
        log(f"[SHOT {shot_num}] ❌ Unknown processing_type: {processing_type}")
        return None

# ========================================
# STEP 12: PROCESS COMPLETE QUESTION
# ========================================

def process_question(forecast: dict, veo_count: dict) -> Optional[List[str]]:
    """
    Process complete question: generate all shots + ADR audio.
    
    Returns: List of *_PROCESSED.mp4 file paths, or None if failed
    """
    q_id = forecast.get("metadata", {}).get("question_id", "unknown")
    
    log_progress(f"🎬 STARTING STAGE 4 FOR Q {q_id}")
    
    # Step 1: Generate script + dialogue_only
    result = generate_script(forecast)
    if not result:
        return None
    
    script, dialogue_only = result
    
    # Step 2: Generate Veo + ElevenLabs prompts
    combined_prompts = generate_veo_elevenlabs_prompts(script)
    if not combined_prompts:
        return None
    
    # NEW: Check if we should stop after scripts
    if STOP_AFTER_VEO_SCRIPTS:
        log_progress(f"⏸️ STAGE 4 STOPPED AFTER SCRIPTS FOR Q {q_id} (stop_after_veo_scripts=true)")
        return [f"scripts_only_q{q_id}"]
    
    # Step 3: Load anchor image (once per question)
    anchor_image = load_anchor_image()
    if not anchor_image:
        log("[VEO] ⚠️ No anchor image loaded - anchor shots may be inconsistent")
    
    # Step 4: Process each shot
    shots = combined_prompts.get("shots", [])
    processed_files = []
    
    for shot_data in shots:
        processed_path = process_shot(shot_data, anchor_image, q_id, veo_count)
        
        if processed_path:
            processed_files.append(processed_path)
        else:
            shot_num = shot_data.get("shot", "?")
            
            # Check if we hit rate limit
            if MAX_VEO_GENERATIONS_PER_RUN and veo_count["count"] >= MAX_VEO_GENERATIONS_PER_RUN:
                log(f"[Q {q_id}] ⏸️ RATE LIMIT REACHED at shot {shot_num}")
                log(f"[Q {q_id}] 💡 Run pipeline again to continue from shot {shot_num}")
                break
            else:
                log(f"[Q {q_id}] ⚠️ Shot {shot_num} failed - continuing with next shot")
    
    # Step 5: Generate ADR audio (if we completed any shots)
    if processed_files:
        generate_adr_audio(dialogue_only, q_id)
    
    if not processed_files:
        log(f"[Q {q_id}] ❌ No shots processed successfully")
        return None
    
    log_progress(f"✅ Q {q_id} COMPLETE: {len(processed_files)}/{len(shots)} shots processed")
    
    return processed_files

# ========================================
# STEP 13: FINAL MASTER VIDEO
# ========================================

def create_master_video(all_processed_files: List[str]) -> Optional[str]:
    """
    Concatenate all *_PROCESSED.mp4 files into final master video.
    
    Returns: Path to master video, or None if failed
    """
    if not all_processed_files:
        log("[MASTER] ❌ No processed files to concatenate")
        return None
    
    log_progress(f"🎬 CREATING MASTER VIDEO from {len(all_processed_files)} clips")
    
    try:
        # Create concat list file
        concat_list_path = os.path.join("out", "concat_list.txt")
        with open(concat_list_path, 'w') as f:
            for file_path in all_processed_files:
                # Use just filename (all files are in out/ directory)
                filename = os.path.basename(file_path)
                f.write(f"file '{filename}'\n")
        
        log(f"[MASTER] 📝 Created concat list with {len(all_processed_files)} files")
        
        # Run ffmpeg
        output_filename = "final_master_video.mp4"
        output_path = os.path.join("out", output_filename)
        
        result = subprocess.run([
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', 'concat_list.txt',
            '-c', 'copy',
            '-y',  # Overwrite output
            output_filename
        ], cwd='out', capture_output=True, text=True)
        
        if result.returncode != 0:
            log(f"[MASTER] ❌ ffmpeg failed: {result.stderr}")
            return None
        
        log(f"[MASTER] ✅ Master video saved to {output_path}")
        
        # Clean up concat list
        try:
            os.remove(concat_list_path)
        except:
            pass
        
        return output_path
        
    except Exception as e:
        log(f"[MASTER] ❌ Master video creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# ========================================
# MAIN: RUN STAGE 4 FOR ALL FORECASTS
# ========================================

def run_stage4(forecasts: List[dict]) -> Optional[Dict[str, any]]:
    """
    Run Stage 4 for all forecasts sequentially.
    
    Returns: Dict with results or None if failed
    """
    log_progress("🎬 STARTING STAGE 4: VEO + ELEVENLABS VIDEO GENERATION")
    
    # Reset to root directory
    set_current_question(None)
    
    # Validate configuration
    if not PROMPT_REPORTER1 or not PROMPT_REPORTER2:
        log("[STAGE4] ⏭️ SKIPPED (PROMPT_REPORTER1 or PROMPT_REPORTER2 not configured)")
        return {"skipped": "Prompts not configured"}
    
    if not STOP_AFTER_VEO_SCRIPTS and not GEMINI_API_KEY:
        log("[STAGE4] ⏭️ SKIPPED (GEMINI_API_KEY not configured)")
        return {"skipped": "Gemini API key not configured"}
    
    if not STOP_AFTER_VEO_SCRIPTS and not ELEVENLABS_API_KEY:
        log("[STAGE4] ⏭️ SKIPPED (ELEVENLABS_API_KEY not configured)")
        return {"skipped": "ElevenLabs API key not configured"}
    
    if STOP_AFTER_VEO_SCRIPTS:
        log("[STAGE4] 📝 SCRIPTS ONLY MODE - Will stop after generating prompts")
    
    if not forecasts:
        log("[STAGE4] ⚠️ No forecasts provided")
        return None
    
    # Initialize video generation counter (shared between Veo and Sora2)
    veo_count = {"count": 0}
    
    # Process each question
    all_processed_files = []
    scripts_generated = []
    
    for forecast in forecasts:
        q_id = forecast.get("metadata", {}).get("question_id", "unknown")
        
        result = process_question(forecast, veo_count)
        
        if result:
            if STOP_AFTER_VEO_SCRIPTS:
                # Scripts-only mode
                scripts_generated.append(q_id)
            else:
                # Video mode - collect processed files
                all_processed_files.extend(result)
                log(f"[STAGE4] ✅ Q {q_id}: {len(result)} shots completed")
        else:
            log(f"[STAGE4] ⚠️ Q {q_id} failed or incomplete")
        
        # Check if we hit rate limit
        if MAX_VEO_GENERATIONS_PER_RUN and veo_count["count"] >= MAX_VEO_GENERATIONS_PER_RUN:
            log(f"\n{'='*80}")
            log(f"[STAGE4] ⏸️ RATE LIMIT REACHED: {veo_count['count']}/{MAX_VEO_GENERATIONS_PER_RUN} videos generated")
            log(f"[STAGE4] 💡 THIS RUN IS STOPPING HERE")
            log(f"[STAGE4] 🔄 Run the pipeline again to generate more videos")
            log(f"[STAGE4] 📊 Progress: {len(all_processed_files)} shots completed across {len(forecasts)} questions")
            log(f"{'='*80}\n")
            break
    
    # Handle scripts-only mode
    if STOP_AFTER_VEO_SCRIPTS:
        if scripts_generated:
            log_progress(f"✅ STAGE 4 COMPLETE (SCRIPTS ONLY): {len(scripts_generated)} scripts generated")
            return {q_id: f"scripts_only_{q_id}" for q_id in scripts_generated}
        else:
            return None
    
    # Handle video mode - create master video
    if not all_processed_files:
        log("[STAGE4] ❌ No processed files - cannot create master video")
        return None
    
    log_progress(f"🎬 CREATING MASTER VIDEO: {len(all_processed_files)} clips across {len(forecasts)} questions")
    
    master_video = create_master_video(all_processed_files)
    
    if master_video:
        log_progress(f"✅ STAGE 4 COMPLETE: Master video created with {len(all_processed_files)} shots")
        return {
            "master_video": master_video,
            "total_shots": len(all_processed_files),
            "questions_processed": len([f for f in forecasts if any(q_id in pf for pf in all_processed_files for q_id in [f.get("metadata", {}).get("question_id", "")])]),
            "veo_generations": veo_count["count"]
        }
    else:
        log("[STAGE4] ❌ Failed to create master video")
        return {
            "master_video": None,
            "total_shots": len(all_processed_files),
            "processed_files": all_processed_files,
            "veo_generations": veo_count["count"]
        }

# ========================================
# END OF STAGE4_VEO
# ========================================
