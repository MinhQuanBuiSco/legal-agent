import logging

from datasets import load_dataset

from legal_agent.data_pipeline.sentence_tokenization import \
    document_segmentation
from legal_agent.data_pipeline.text_processing import normalize_text
from legal_agent.nlp.embeddings import get_embedding
from legal_agent.nlp.entity_extraction import get_ner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def chunk_document(examples):
    chunks = []
    ids = []
    documents = []
    titles = []
    states = []
    issuers = []
    timestamps = []

    for i, sentence in enumerate(examples["document"]):
        sentence_chunks = document_segmentation(sentence)

        # Duplicate other column values for each chunk
        ids.extend([examples["id"][i]] * len(sentence_chunks))
        documents.extend([examples["document"][i]] * len(sentence_chunks))
        titles.extend([examples["title"][i]] * len(sentence_chunks))
        states.extend([examples["state"][i]] * len(sentence_chunks))
        issuers.extend([examples["issuer"][i]] * len(sentence_chunks))
        timestamps.extend([examples["timestamp"][i]] * len(sentence_chunks))
        chunks.extend(sentence_chunks)

    return {"sentence": chunks, "id": ids, "document": documents, "title": titles, "state": states, "issuer": issuers, "timestamp": timestamps}


def process_dataset():
    # Load the entire US dataset
    dataset = load_dataset("HFforLegal/case-law", split="us[:500]", verification_mode="no_checks")
    # Remove unnecessary columns
    dataset = dataset.remove_columns(["citation", "docket_number","hash"])
    # Normalize text
    dataset = dataset.map(lambda x: {"document": normalize_text(x["document"])}, num_proc=10)
    print(dataset)
    # Segmentation Expand rows
    dataset = dataset.map(chunk_document, batched=True)
    # NER
    dataset = dataset.map(lambda x: {"entities": " , ".join([i["word"].strip().lower() for i in get_ner(x["sentence"])])})
    # Embeddings
    dataset = dataset.map(lambda x: {"embedding": get_embedding(x["sentence"])})
    print(dataset[:5])
    # Save the processed dataset
    dataset.save_to_disk("src/legal_agent/database/")


if __name__ == "__main__":
    process_dataset()
