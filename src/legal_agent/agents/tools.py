import os

from datasets import load_from_disk
from smolagents import Tool

from legal_agent.nlp.embeddings import get_embedding, legal_tokenizer
from legal_agent.nlp.summarization import caselaw_sumarization


class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        default_database_dir = os.path.join(base_dir, "../database/")
        self.database = load_from_disk(default_database_dir)
        self.database.add_faiss_index(column="embedding")

    def forward(self, query: str) -> str:
        query_length = len(legal_tokenizer(query, add_special_tokens=False)["input_ids"])
        print(query_length)

        if query_length > 512:
            print("Query too long, summarizing...")
            query = caselaw_sumarization(query)
            print(query)
        query_embeddings = get_embedding(query)
        scores, retrieved_examples = self.database.get_nearest_examples("embedding", query_embeddings, k=1)
        document = retrieved_examples["document"][0]
        return f"""
        Similar documents found in the database:
        {document}
"""
