## Installation

To run the code in this project, first, create a Python virtual environment using e.g. `uv`.
To install `uv`, follow the [UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/).

## Install environment and build legal database

```shell
bash scripts/build_database.sh
```
## Activate the environment

```shell
source .venv/bin/activate
```
## Huggingface Token and Open API Key
Please add the Hugging Face token and OpenAI API key in the .env_example file, then rename it to .env.

## Run Agent
```
python src/legal_agent/agents/rag_agent.py
```
## Limitation 
1. The database contains only 500 samples, which limits the effectiveness of retrieval.
2. The prompt requires refinement, as the GPT-4o model occasionally alters the query.
3. The setup has been tested on a Mac M4, which may lead to issues on Linux or Windows, such as missing CUDA support or the model failing to load onto the GPU.