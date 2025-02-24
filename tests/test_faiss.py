from legal_agent.nlp.embeddings import get_embedding
from datasets import load_from_disk

def test_get_embedding():
    database = load_from_disk("src/legal_agent/database/")
    database.add_faiss_index(column='embedding')
    question = "What is the legal status of marijuana in the United States?"
    question_embedding = get_embedding(question)
    scores, retrieved_examples = database.get_nearest_examples('embedding', question_embedding, k=10)
    assert len(retrieved_examples["sentence"][0]) > 0
    assert len(scores) == 10