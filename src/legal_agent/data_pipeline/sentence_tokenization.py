import logging

import nltk
# Load English NLP model
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

nltk.download("punkt_tab")
tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def document_segmentation(sentence: str, max_tokens: int = 500) -> list:
    """
    Splits a long sentence into smaller chunks that fit within max_tokens.

    Args:
        sentence: A long sentence string.
        max_tokens: The maximum token length for each chunk.

    Returns:
        A list of sentence chunks.
    """
    tokenized = tokenizer(sentence, add_special_tokens=False)
    input_ids = tokenized["input_ids"]
    chunks = []
    for i in range(0, len(input_ids), max_tokens):
        sub_chunk_ids = input_ids[i : i + max_tokens]
        chunk_text = tokenizer.decode(sub_chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks
