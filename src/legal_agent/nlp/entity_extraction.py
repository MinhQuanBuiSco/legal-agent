from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          pipeline)

from legal_agent.utils.config_loader import config

model_name = config["model"]["ner_model"]
ner_tokenizer = AutoTokenizer.from_pretrained(model_name)
ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")


# Process text sample (from wikipedia)
def get_ner(text: str) -> dict:
    """Perform named entity recognition on text."""
    ner_result = ner_pipeline(text)
    return ner_result
