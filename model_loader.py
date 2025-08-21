"""
model_loader.py
----------------
Utilities to load a small Hugging Face text-generation model locally.
GPU is optional; if available, it will be used automatically.
"""
from __future__ import annotations

import os
from typing import Optional, Dict, Any

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

# ✅ Default to Flan-T5 for factual Q&A
DEFAULT_MODEL = os.getenv("HF_MODEL_NAME", "google/flan-t5-large")


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL,
    dtype: Optional[str] = None,
    device_preference: Optional[int] = None,
    trust_remote_code: bool = False,
) -> Dict[str, Any]:
    """
    Load tokenizer and model, then create the correct pipeline.
    Returns a dict with {"tokenizer", "model", "pipe"}.
    """
    # Pick dtype
    torch_dtype = None
    if dtype:
        dtype = dtype.lower()
        if dtype in ("fp16", "float16", "half"):
            torch_dtype = torch.float16
        elif dtype in ("bf16", "bfloat16"):
            torch_dtype = torch.bfloat16
        elif dtype in ("fp32", "float32", "full"):
            torch_dtype = torch.float32

    # Decide device
    if device_preference is not None:
        device = device_preference
    else:
        device = 0 if torch.cuda.is_available() else -1

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pick correct model class
    if any(x in model_name.lower() for x in ["t5", "flan", "bart", "mbart", "pegasus", "marian"]):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        task = "text2text-generation"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        task = "text-generation"

    # Build pipeline
    pipe = pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    return {"tokenizer": tokenizer, "model": model, "pipe": pipe}
