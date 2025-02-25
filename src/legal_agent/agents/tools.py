from datasets import load_from_disk
import os
from smolagents import Tool, LiteLLMModel, CodeAgent
from dotenv import load_dotenv
from legal_agent.nlp.embeddings import get_embedding, legal_tokenizer
from legal_agent.nlp.summarization import caselaw_sumarization
from legal_agent.data_pipeline.text_processing import normalize_text
from legal_agent.utils.config_loader import config

load_dotenv()

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
    
    def __init__(self, database_path: str = "src/legal_agent/database/", **kwargs):
        super().__init__(**kwargs)
        self.database = load_from_disk(database_path)
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

document = """
    Please find the relevant document for the case below:
    MISCELLANEOUS SUPREME COURT DISPOSITIONS BALLOT TITLE CERTIFIED November 10, 2011 Satrum v. Kroger (S059691). Petitioner's request for oral argument is denied. Petitioner's argument that the Attorney General's certified ballot title for Initiative Petition No. 20 (2012) does not comply substantially with ORS 250.035(2) to (6) is not well taken. The court certifies to the Secretary of State the Attorney General's certified ballot title for the proposed ballot measure. 
"""

retriever_tool = RetrieverTool()

agent = CodeAgent(
    tools=[retriever_tool],
    model=LiteLLMModel(model_id="gpt-4o", api_key=os.getenv("OPENAI_API_KEY")),
    max_steps=4,
    verbosity_level=2,
)

agent.prompt_templates["system_prompt"] = config["prompts"]["system_prompt"]
document = normalize_text(document)
agent_output = agent.run(document)

print("Final output:")
print(agent_output)