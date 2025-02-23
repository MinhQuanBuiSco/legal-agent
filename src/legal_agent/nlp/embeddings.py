import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# Load tokenizer and model
model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()  # Set model to evaluation mode

# Load dataset (100 examples from 'crime_and_punish')

def compute_embedding(text: str) -> np.ndarray:
    """
    Computes the Legal-BERT embedding for a given text.
    
    Args:
        example (dict): Dictionary containing a "line" key (text data).
    
    Returns:
        dict: A dictionary containing the computed embedding as a NumPy array.
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    # Forward pass without gradients
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract CLS embedding (sentence representation)
    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()  # Shape: (768,)

    return cls_embedding