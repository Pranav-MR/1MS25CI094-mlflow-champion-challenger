"""
Model loader module for Champion and Challenger Hugging Face models.
Provides model configuration, tokenizer and model loading for CPU inference.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_CONFIGS = {
    "model_a": {
        "hf_model_id": "distilbert-base-uncased-finetuned-sst-2-english",
        "role": "Champion",
        "run_name": "A_sentiment_distilbert-sst2",
        "display_name": "DistilBERT (Champion)",
        "output_predictions_file": "outputs/predictions_distilbert.csv",
        "confusion_matrix_file": "outputs/confusion_matrix_distilbert.png",
    },
    "model_b": {
        "hf_model_id": "textattack/bert-base-uncased-SST-2",
        "role": "Challenger",
        "run_name": "B_sentiment_bert-sst2",
        "display_name": "BERT (Challenger)",
        "output_predictions_file": "outputs/predictions_bert.csv",
        "confusion_matrix_file": "outputs/confusion_matrix_bert.png",
    },
}


def load_tokenizer_and_model(hf_model_id: str, device: str = "cpu"):
    """
    Load Hugging Face tokenizer and sequence classification model.
    Puts model in evaluation mode on the specified device (default: CPU).
    """
    print(f"[ModelLoader] Loading tokenizer and model for: {hf_model_id} on {device}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        model = AutoModelForSequenceClassification.from_pretrained(hf_model_id)
        model.to(device)
        model.eval()
        return tokenizer, model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Hugging Face model or tokenizer '{hf_model_id}'.\n"
            f"Original error: {str(e)}"
        )
