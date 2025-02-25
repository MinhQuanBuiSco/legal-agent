#!/bin/bash
make venv

make install

source .venv/bin/activate

python src/legal_agent/data_pipeline/pipeline.py