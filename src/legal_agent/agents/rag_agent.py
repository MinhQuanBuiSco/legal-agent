import os

from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel

from legal_agent.agents.tools import RetrieverTool
from legal_agent.data_pipeline.text_processing import normalize_text
from legal_agent.utils.config_loader import config

load_dotenv()

fail_test_case = """
    Please find the relevant document for the case below:
    MISCELLANEOUS SUPREME COURT DISPOSITIONS BALLOT TITLE CERTIFIED November 10, 2011 Satrum v. Kroger (S059691). Petitioner's request for oral argument is denied. Petitioner's argument that the Attorney General's certified ballot title for Initiative Petition No. 20 (2012) does not comply substantially with ORS 250.035(2) to (6) is not well taken. The court certifies to the Secretary of State the Attorney General's certified ballot title for the proposed ballot measure."""

success_test_case = """
    Please find the relevant document for the case below:
    No. 29412 IN THE SUPREME COURT OF THE STATE OF HAWAI'I IN RE: CRAIG S. HARRISON, Petitioner. ORIGINAL PROCEEDING ORDER DENYING PETITION TO RESIGN AND SURRENDER LICENSE : Nakayama, Acoba, and Duffy, JJ.) Moon, C.J., Levinson, Upon consideration of Petitioner Craig S. Harrison’s Petition to Resign and Surrender License, the attached affidavits, and the lack of objections by the Office of Disciplinary Counsel, it appears that the Petitioner Harrison's motion to waive the $125.00 filing fee required by RSCH 1.10(b) was denied on December 16, 2008, and Petitioner Harrison has not paid the $125.00 filing fee. Therefore, IT 18 HEREBY ORDERED that the petition is denied Hawai'l, December 22, 2008. DATED: Honolulu, COsKY 22 930 eo aad """

retriever_tool = RetrieverTool()

agent = CodeAgent(
    tools=[retriever_tool],
    model=LiteLLMModel(model_id="gpt-4o", api_key=os.getenv("OPENAI_API_KEY")),
    max_steps=4,
    verbosity_level=2,
)

agent.prompt_templates["system_prompt"] = config["prompts"]["system_prompt"]
document = normalize_text(fail_test_case)
agent_output = agent.run(document)

print("Final output:")
print(agent_output)
