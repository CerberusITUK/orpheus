# Rebuild trigger: v3
import runpod
import torch
import base64
import io
import wave
import numpy as np
import uuid
import os
from huggingface_hub import login
from orpheus_tts import OrpheusModel

# Patch vllm tokenizer bug with newer tokenizers library
try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    _orig_getattr = PreTrainedTokenizerBase.__getattr__
    def _patched_getattr(self, key):
        if key == 'all_special_tokens_extended':
            return []
        return _orig_getattr(self, key)
    PreTrainedTokenizerBase.__getattr__ = _patched_getattr
except Exception:
    pass

# Authenticate with HuggingFace if token is provided
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

# Lazy-load models on first use to reduce VRAM footprint
_model = None
_clone_model = None

def get_model():
    global _model
    if _model is None:
        _model = OrpheusModel(
            model_name="canopylabs/orpheus-tts-0.1-finetune-prod"
        )
    return _model

def get_clone_model():
    global _clone_model
    if _clone_model is None:
        _clone_model = OrpheusModel(
            model_name="canopylabs/orpheus-3b-0.1-pretrained"
        )
    return _clone_model

def handler(event):
    """
    RunPod serverless handler for Orpheus TTS.
    
    Input fields:
    - text (string, required) — the story text, may include <gasp> <sigh> etc tags
    - voice (string, optional, default "dan") — preset voice name
    - reference_audio (string, optional) — base64-encoded WAV for voice cloning (not yet implemented)
    - reference_text (string, optional) — transcript of reference audio (not yet implemented)
    - temperature (float, optional, default 0.7)
    - repetition_penalty (float, optional, default 1.1)
    
    Returns:
    - audio_base64 (string) — base64-encoded WAV audio (mono, 16-bit, 24000Hz)
    - duration_seconds (float)
    - chunks_generated (int)
    """
    input_data = event["input"]
    
    # Extract parameters with defaults
    text = input_data.get("text")
    if not text:
        raise ValueError("text is required")
    
    voice = input_data.get("voice", "dan")
    reference_audio_b64 = input_data.get("reference_audio")
    reference_text = input_data.get("reference_text")
    temperature = float(input_data.get("temperature", 0.7))
    repetition_penalty = float(input_data.get("repetition_penalty", 1.1))
    
    chunks_generated = 0
    all_audio_bytes = bytearray()
    ref_path = None
    
    # Handle reference audio for voice cloning
    if reference_audio_b64:
        ref_filename = f"ref_{uuid.uuid4().hex}.wav"
        ref_path = os.path.join("/tmp", ref_filename)
        ref_bytes = base64.b64decode(reference_audio_b64)
        with open(ref_path, "wb") as f:
            f.write(ref_bytes)
    
    # Split long text into chunks at sentence boundaries
    if len(text) > 1500:
        chunks = []
        current_chunk = ""
        sentences = []
        
        # Split into sentences by punctuation
        text_remaining = text
        for delimiter in [". ", "! ", "? "]:
            if delimiter in text_remaining:
                parts = text_remaining.split(delimiter)
                for i, part in enumerate(parts):
                    if i < len(parts) - 1:
                        sentences.append(part + delimiter)
                    else:
                        text_remaining = part
                break
        
        if text_remaining:
            sentences.append(text_remaining)
        
        # Accumulate sentences into chunks under 1500 chars
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= 1500:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        if not chunks:
            chunks = [text]
    else:
        chunks = [text]
    
    # Generate audio for each chunk
    for chunk in chunks:
        if ref_path:
            # Use pretrained model with voice cloning
            audio_chunks_generator = get_clone_model().generate_speech(
                prompt=chunk,
                voice=voice,
                temperature=temperature,
                repetition_penalty=repetition_penalty
            )
        else:
            # Use finetuned model with preset voice
            audio_chunks_generator = get_model().generate_speech(
                prompt=chunk,
                voice=voice,
                temperature=temperature,
                repetition_penalty=repetition_penalty
            )
        
        # Collect all yielded bytes and append
        for audio_chunk in audio_chunks_generator:
            if isinstance(audio_chunk, bytes):
                all_audio_bytes.extend(audio_chunk)
            elif isinstance(audio_chunk, bytearray):
                all_audio_bytes.extend(audio_chunk)
            elif isinstance(audio_chunk, str):
                all_audio_bytes.extend(audio_chunk.encode('utf-8'))
        
        chunks_generated += 1
    
    # Clean up temp reference audio file
    if ref_path and os.path.exists(ref_path):
        os.remove(ref_path)
    
    # Encode to base64
    audio_base64 = base64.b64encode(all_audio_bytes).decode('utf-8')
    
    # Estimate duration (approx 24kHz, 16-bit mono)
    sample_rate = 24000
    bytes_per_sample = 2
    duration_seconds = len(all_audio_bytes) / (sample_rate * bytes_per_sample)
    
    return {
        "audio_base64": audio_base64,
        "duration_seconds": round(duration_seconds, 2),
        "chunks_generated": chunks_generated
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
