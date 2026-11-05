"""
Shared configuration for the EduAssist LLMOps assignment: model choice, pricing
reference, the five academic questions, and the three prompt template versions.

Model note
----------
All LLM calls in this project run against ``Qwen/Qwen2.5-3B-Instruct`` (Apache-2.0,
~3B parameters, strong instruction following), loaded locally via Hugging Face
``transformers``. Token counts come from this model's own tokenizer.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

# Model executed locally for every API call in this notebook.
MODEL_ID_LOCAL = "Qwen/Qwen2.5-3B-Instruct"

# Hugging Face token, if the grader wants to re-run this against the real
# Inference API / router instead of the local fallback. Must be supplied via
# environment variable - never hard-code a token.
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip() or None

# --------------------------------------------------------------------------
# Pricing (Task 2) - published, dated, with explicit currency.
# --------------------------------------------------------------------------
# Source: OpenRouter's aggregated model-pricing API (https://openrouter.ai/api/v1/models),
# entry id "qwen/qwen-2.5-7b-instruct", accessed 2026-11-04. Qwen2.5-7B-Instruct is the
# smallest Qwen2.5 checkpoint currently listed by hosted providers (the 3B size isn't
# separately listed), so its published rate is used as the same-family reference price.
# Raw rates were $0.0000001 / input token and $0.0000002 / output token.
PRICING = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "pricing_date": "2026-11-04",
    "currency": "USD",
    "input_price_per_million": 0.10,
    "output_price_per_million": 0.20,
    "source_url": "https://openrouter.ai/api/v1/models",
}

MONTHLY_PROJECTED_REQUESTS = 10_000

# --------------------------------------------------------------------------
# Generation settings
# --------------------------------------------------------------------------
MAX_NEW_TOKENS = 300
# Greedy decoding keeps latency/quality comparisons reproducible across prompt
# versions (removes sampling variance as a confound between v1/v2/v3).
GENERATION_KWARGS = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": False,
    "temperature": None,
    "top_p": None,
}

# --------------------------------------------------------------------------
# The five academic questions (Task 1-3 all reuse this fixed set)
# --------------------------------------------------------------------------
QUESTIONS = [
    {
        "question_id": "Q1",
        "topic": "Machine Learning",
        "question": "What is the difference between supervised and unsupervised learning?",
    },
    {
        "question_id": "Q2",
        "topic": "Physics",
        "question": "Explain Newton's second law of motion with an example.",
    },
    {
        "question_id": "Q3",
        "topic": "Algorithms",
        "question": "What is the time complexity of binary search and why?",
    },
    {
        "question_id": "Q4",
        "topic": "Biology",
        "question": "Describe the process of photosynthesis in plants.",
    },
    {
        "question_id": "Q5",
        "topic": "Statistics",
        "question": "What is the Central Limit Theorem in statistics?",
    },
]

# --------------------------------------------------------------------------
# Prompt versions (Task 3) - verbatim from the assignment, using MLflow's
# native double-curly-brace template syntax (mlflow.register_prompt /
# Prompt.format both operate on {{variable}} placeholders).
# --------------------------------------------------------------------------
PROMPT_TEMPLATES = {
    "v1": {
        "label": "Basic",
        "commit_message": "v1: basic direct-answer prompt, no role or formatting guidance",
        "template": (
            "Answer the following student question clearly.\n"
            "Question: {{question}}"
        ),
    },
    "v2": {
        "label": "Structured",
        "commit_message": "v2: adds academic-assistant persona and undergraduate tone guidance",
        "template": (
            "You are an academic assistant.\n"
            "Answer the following student question in a clear, concise, and accurate manner.\n"
            "Use simple language suitable for undergraduate students.\n"
            "Question: {{question}}"
        ),
    },
    "v3": {
        "label": "JSON Output",
        "commit_message": "v3: enforces strict JSON schema output (answer/confidence/key_points)",
        "template": (
            "You are an academic assistant.\n"
            "Answer the student question and return ONLY valid JSON.\n"
            "Do not include markdown, explanations outside JSON, or code fences.\n"
            "\n"
            "Use this exact JSON schema:\n"
            "{\n"
            '  "answer": "string",\n'
            '  "confidence": "high | medium | low",\n'
            '  "key_points": ["string"]\n'
            "}\n"
            "Question: {{question}}"
        ),
    },
}

MLFLOW_EXPERIMENT_NAME = "eduassist_llmops_assignment"
PRODUCTION_ALIAS = "production"
