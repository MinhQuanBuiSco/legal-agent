from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          pipeline)

ner_tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
ner_model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")


# Process text sample (from wikipedia)
def get_ner(text: str) -> dict:
    """Perform named entity recognition on text."""
    ner_result = ner_pipeline(text)
    return ner_result
