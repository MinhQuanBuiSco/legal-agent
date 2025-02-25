from datasets import load_from_disk


def test_filter_timestamp():
    database = load_from_disk("src/legal_agent/database/")
    test_timestamp = "2015-02-10T00:00:00Z"
    results = database.filter(lambda x: x["timestamp"] > test_timestamp)
    assert len(results["timestamp"]) > 0
