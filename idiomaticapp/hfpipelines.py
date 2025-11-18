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

BASE_DIR = "/models/Llama-3.2-1B-Instruct"
SNAPSHOT_SHA = os.listdir(os.path.join(BASE_DIR, "snapshots"))[0]
MODEL_DIR = os.path.join(BASE_DIR, "snapshots", SNAPSHOT_SHA)


class IdiomaticIvan:
    _pipeline: TextGenerationPipeline

    MESSAGES = [
         {
            "role": "system",
            "content": """You are a friendly chatbot named Ivan who always responds in casual style
            in language that uses a healthy mixture idioms from literature. and popular idioms of the English language used in any dialect of the
            United States""",
        },
    ]

    def __init__(self):
        self._set_up_pipeline()

    def _set_up_pipeline(self):
        # Load tokenizer and model from local directory
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            local_files_only=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            local_files_only=True
        )

        # Create the pipeline
        self._pipeline = TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=0  # GPU; use device=-1 or omit for CPU
        )

    def prompt(self, prompt: str) -> str:
        generated_text = self._pipeline(self.MESSAGES + [{"role": "user", "content": prompt}], max_new_tokens=512)[0]['generated_text']
        assistant_response = list(filter(lambda x: x['role'] == 'assistant', generated_text))[0]['content']
        return assistant_response