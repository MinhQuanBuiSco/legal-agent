from legal_agent.nlp.entity_extraction import get_ner
from legal_agent.nlp.embeddings import get_embedding

def test_get_ner():
    text = "Apple Inc. is an American multinational technology company that specializes in consumer electronics, computer software, and online services. Apple is the world's largest technology company by revenue and, since January 2021, the world's most valuable company."
    entities = get_ner(text)
    assert len(entities) == 4
    assert entities[0]["word"].strip() == "Apple Inc."
    assert entities[0]["entity_group"] == "ORG"
    assert entities[0]["start"] == 0
    assert entities[0]["end"] == 10

def test_get_embeddings():
    text = "Apple Inc. is an American multinational technology company that specializes in consumer electronics, computer software, and online services. Apple is the world's largest technology company by revenue and, since January 2021, the world's most valuable company."
    embedding = get_embedding(text)
    assert embedding.shape == (768,)