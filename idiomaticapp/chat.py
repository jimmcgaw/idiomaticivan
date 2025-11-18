import os

import torch

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
    # TextClassificationPipeline,
)

# _prompt_guard_pipeline: TextClassificationPipeline = pipeline("text-classification", model="meta-llama/Prompt-Guard-86M")

BASE_DIR = "/Users/jimmcgaw/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct"
SNAPSHOT_SHA = os.listdir(os.path.join(BASE_DIR, "snapshots"))[0]
MODEL_DIR = os.path.join(BASE_DIR, "snapshots", SNAPSHOT_SHA)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

# 2. Define a (simple) chat template
#
# This is a Jinja2-style template; `messages` is a list of
# {"role": "system"|"user"|"assistant", "content": "..."} dicts.
tokenizer.chat_template = r"""
{% for message in messages %}
{% if message['role'] == 'system' %}
[SYS] {{ message['content'] }} [/SYS]
{% elif message['role'] == 'user' %}
User: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}
Assistant: {{ message['content'] }}
{% endif %}
{% endfor %}
Assistant:
""".strip()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
)

pipeline = TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    device=0  # GPU; use device=-1 or omit for CPU
)

# 4. Helper that takes chat-style messages and uses the template
def chat(messages, **gen_kwargs) -> str:
    """
    messages: list[dict] like
        {"role": "user"|"system"|"assistant", "content": "text"}
    gen_kwargs: extra args to pipeline (max_new_tokens, temperature, etc.)
    """
    # Build the text prompt using the tokenizer's chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,              # we want a string, not token IDs
        add_generation_prompt=True,  # adds the final "Assistant:" or similar
    )

    # Call the TextGenerationPipeline
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,   # only generated completion, not the prompt
        **gen_kwargs,
    )

    # The pipeline returns a list of dicts: [{"generated_text": "..."}]
    return outputs[0]["generated_text"]


messages = [
    {
        "role": "system",
        "content": """You are a friendly chatbot named Ivan who always responds in casual style
        in language that uses a healthy mixture of popular idioms of the English language used in any dialect of the
        United States and idioms from literature.""",
    },
    {"role": "user", "content": "What are some things people do for fun when they visit San Francisco?"},
]

reply = chat(messages)
print("Assistant:", reply)
