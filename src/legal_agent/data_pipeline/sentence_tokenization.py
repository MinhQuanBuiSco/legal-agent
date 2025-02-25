import logging

from transformers import AutoTokenizer
from legal_agent.utils.config_loader import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ner_tokenizer = AutoTokenizer.from_pretrained(config["model"]["embedding_model"])

def document_segmentation(sentence: str, max_tokens: int = 500) -> list:
    """
    Splits a long sentence into smaller chunks that fit within max_tokens.

    Args:
        sentence: A long sentence string.
        max_tokens: The maximum token length for each chunk.

    Returns:
        A list of sentence chunks.
    """
    tokenized = ner_tokenizer(sentence, add_special_tokens=False)
    input_ids = tokenized["input_ids"]
    chunks = []
    for i in range(0, len(input_ids), max_tokens):
        sub_chunk_ids = input_ids[i : i + max_tokens]
        chunk_text = ner_tokenizer.decode(sub_chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks
