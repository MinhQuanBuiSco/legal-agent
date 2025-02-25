## Installation

To run the code in this project, first, create a Python virtual environment using e.g. `uv`.
To install `uv`, follow the [UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/).

# Install environment and build legal database

```shell
bash scripts/build_database.sh
```
# Activate the environment

```shell
source .venv/bin/activate
```
# Huggingface Token and Open API Key
Please add the Hugging Face token and OpenAI API key in the .env_example file, then rename it to .env.

# Run Agent
python src/legal_agent/agents/rag_agent.py