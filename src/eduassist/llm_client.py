"""
LLM call wrapper used by every task in the notebook.

Two call paths are implemented:

1. ``call_via_hf_inference_api`` - the "real API" path (Hugging Face Inference
   Providers router, OpenAI-compatible). Only used when ``HF_TOKEN`` is set in
   the environment. Reports token usage exactly as returned by the API
   (consistent with "LLM APIs commonly bill separately for input and output
   tokens, and usage details are available in API responses").
2. ``call_via_local_model`` - the documented fallback used in this submission
   (no HF_TOKEN is configured here). Loads ``Qwen/Qwen2.5-3B-Instruct`` once
   with Hugging Face `transformers` and generates locally on MPS/CPU. Token
   counts come from the same tokenizer that encodes/decodes the request, so
   they are exact, not estimates.

``call_llm(...)`` is the single entry point the notebook uses; it picks the
path automatically based on whether HF_TOKEN is present.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from eduassist.config import GENERATION_KWARGS, HF_TOKEN, MODEL_ID_LOCAL

HF_ROUTER_CHAT_COMPLETIONS_URL = "https://router.huggingface.co/v1/chat/completions"


@dataclass
class LLMResult:
    response_text: str
    input_tokens: int
    output_tokens: int
    status: str  # "success" | "error"
    error_message: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    latency_s: float = 0.0
    model_name: str = ""
    call_path: str = ""  # "hf_inference_api" | "local_transformers"
    extra: dict = field(default_factory=dict)


class LocalQwenClient:
    """Loads Qwen/Qwen2.5-3B-Instruct once and reuses it for every call."""

    def __init__(self, model_id: str = MODEL_ID_LOCAL):
        self.model_id = model_id
        self._tokenizer = None
        self._model = None
        self._device = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float32 if self._device == "mps" else "auto",
        ).to(self._device)
        self._model.eval()

    def generate(self, prompt_text: str) -> LLMResult:
        import torch

        self._lazy_load()
        messages = [{"role": "user", "content": prompt_text}]
        # transformers>=5 returns a BatchEncoding (dict of tensors) here, not a bare tensor.
        encoded = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self._device)
        input_ids = encoded["input_ids"]
        input_tokens = int(input_ids.shape[1])

        start_time = time.time()
        try:
            with torch.no_grad():
                output_ids = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=encoded.get("attention_mask"),
                    max_new_tokens=GENERATION_KWARGS["max_new_tokens"],
                    do_sample=GENERATION_KWARGS["do_sample"],
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            end_time = time.time()
            new_tokens = output_ids[0][input_tokens:]
            response_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            return LLMResult(
                response_text=response_text,
                input_tokens=input_tokens,
                output_tokens=int(new_tokens.shape[0]),
                status="success",
                start_time=start_time,
                end_time=end_time,
                latency_s=end_time - start_time,
                model_name=self.model_id,
                call_path="local_transformers",
            )
        except Exception as exc:  # pragma: no cover - defensive, surfaced in CSV/report
            end_time = time.time()
            return LLMResult(
                response_text="",
                input_tokens=input_tokens,
                output_tokens=0,
                status="error",
                error_message=str(exc),
                start_time=start_time,
                end_time=end_time,
                latency_s=end_time - start_time,
                model_name=self.model_id,
                call_path="local_transformers",
            )


_LOCAL_CLIENT = LocalQwenClient()


def call_via_hf_inference_api(prompt_text: str, model_id: str = MODEL_ID_LOCAL) -> LLMResult:
    """Real hosted-API path. Requires HF_TOKEN; not exercised in this submission."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": GENERATION_KWARGS["max_new_tokens"],
    }
    start_time = time.time()
    try:
        resp = requests.post(
            HF_ROUTER_CHAT_COMPLETIONS_URL, headers=headers, json=payload, timeout=120
        )
        end_time = time.time()
        resp.raise_for_status()
        body = resp.json()
        usage = body.get("usage", {})
        text = body["choices"][0]["message"]["content"]
        return LLMResult(
            response_text=text,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            status="success",
            start_time=start_time,
            end_time=end_time,
            latency_s=end_time - start_time,
            model_name=model_id,
            call_path="hf_inference_api",
        )
    except Exception as exc:
        end_time = time.time()
        return LLMResult(
            response_text="",
            input_tokens=0,
            output_tokens=0,
            status="error",
            error_message=str(exc),
            start_time=start_time,
            end_time=end_time,
            latency_s=end_time - start_time,
            model_name=model_id,
            call_path="hf_inference_api",
        )


def call_llm(prompt_text: str) -> LLMResult:
    """Single entry point: routes to the hosted API if HF_TOKEN is set, else local."""
    if HF_TOKEN:
        return call_via_hf_inference_api(prompt_text)
    return _LOCAL_CLIENT.generate(prompt_text)
