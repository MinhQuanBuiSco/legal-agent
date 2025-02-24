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

text = "Apple Inc. is an American multinational technology company that specializes in consumer electronics, computer software, and online services. Apple is the world's largest technology company by revenue and, since January 2021, the world's most valuable company."
entities = get_ner(text)
print(entities)