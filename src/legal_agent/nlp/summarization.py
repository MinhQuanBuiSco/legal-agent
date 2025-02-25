import os

import torch
import transformers
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


def caselaw_sumarization(user_input: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a legal AI assistant specializing in case law summarization. Your task is to generate concise, accurate, and structured summaries of court cases. The summaries should highlight key aspects such as case background, legal issues, court decisions, reasoning, and implications. Maintain a professional and neutral tone, ensuring clarity and coherence, and the summary is about 200 words or less.",
        },
        {"role": "user", "content": user_input},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    return outputs[0]["generated_text"][-1]["content"]
