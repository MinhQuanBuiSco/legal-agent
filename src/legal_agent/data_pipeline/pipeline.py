import logging

from datasets import load_dataset

from legal_agent.data_pipeline.sentence_tokenization import \
    document_segmentation
from legal_agent.data_pipeline.text_processing import normalize_text

# from legal_agent.nlp.entity_extraction import perform_ner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def process_dataset():
    # Load the entire US dataset
    dataset = load_dataset("HFforLegal/case-law", split="us[:50000]", verification_mode="no_checks")
    # Normalize text
    dataset = dataset.map(lambda x: {"document": normalize_text(x["document"])}, num_proc=10)
    # Segmentation
    dataset = dataset.map(lambda x: {"document": document_segmentation(x["document"])}, num_proc=10)

    print(dataset)


if __name__ == "__main__":
    process_dataset()
