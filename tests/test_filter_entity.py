from datasets import load_from_disk


def test_get_embedding():
    database = load_from_disk("src/legal_agent/database/")
    test_entites = ["apple", "court"]
    results = database.filter(lambda example: any(word in example["entities"] for word in test_entites))
    print(results)
    assert len(results["timestamp"]) > 0
