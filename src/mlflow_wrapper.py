"""
Custom MLflow pyfunc wrapper for Hugging Face sentiment classification models.
Packages the tokenizer and PyTorch model into a portable mlflow.pyfunc.PythonModel.
"""

import os
import torch
import pandas as pd
import numpy as np
import mlflow.pyfunc
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class HuggingFaceSentimentWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc custom model wrapper for Hugging Face sentiment analysis.
    Satisfies requirements:
    - Loads saved tokenizer and model from artifact directory or model ID
    - Runs CPU inference with PyTorch
    - Returns predictions in a clean pandas DataFrame
    - Self-contained and usable via mlflow.pyfunc.load_model(...)
    """

    def __init__(self, hf_model_id: str = None, max_length: int = 128):
        self.hf_model_id = hf_model_id
        self.max_length = max_length
        self.tokenizer = None
        self.model = None

    def load_context(self, context):
        """
        Called when model is loaded by MLflow.
        Loads the tokenizer and model weights from artifacts.
        """
        model_path = None
        if context.artifacts and "model_dir" in context.artifacts:
            model_path = context.artifacts["model_dir"]
        elif self.hf_model_id:
            model_path = self.hf_model_id
        else:
            raise ValueError("No model_dir artifact or hf_model_id provided to load context.")

        print(f"[MLflowWrapper] Loading tokenizer and model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to("cpu")
        self.model.eval()

    def predict(self, context, model_input):
        """
        Perform sentiment predictions on input texts.
        Accepts:
        - pd.DataFrame with a 'text' or 'sentence' column, or single string column
        - list/Series of string texts

        Returns:
        - pd.DataFrame with ['text', 'prediction', 'sentiment', 'confidence']
        """
        if self.model is None or self.tokenizer is None:
            # Fallback for direct usage without load_context
            if self.hf_model_id:
                self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_id)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.hf_model_id)
                self.model.to("cpu")
                self.model.eval()
            else:
                raise RuntimeError("Model and tokenizer are not initialized.")

        # Extract list of texts from various input types
        if isinstance(model_input, pd.DataFrame):
            if "text" in model_input.columns:
                texts = model_input["text"].astype(str).tolist()
            elif "sentence" in model_input.columns:
                texts = model_input["sentence"].astype(str).tolist()
            elif len(model_input.columns) > 0:
                texts = model_input.iloc[:, 0].astype(str).tolist()
            else:
                texts = []
        elif isinstance(model_input, pd.Series):
            texts = model_input.astype(str).tolist()
        elif isinstance(model_input, (list, tuple)):
            texts = [str(t) for t in model_input]
        elif isinstance(model_input, dict):
            if "text" in model_input:
                val = model_input["text"]
                texts = val if isinstance(val, list) else [val]
            else:
                texts = [str(v) for v in model_input.values()]
        else:
            texts = [str(model_input)]

        if not texts:
            return pd.DataFrame(columns=["text", "prediction", "sentiment", "confidence"])

        # Tokenize batch
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predicted_ids = torch.argmax(probs, dim=-1).cpu().numpy()
            confidences = torch.max(probs, dim=-1).values.cpu().numpy()

        # Map binary label: 0 -> Negative, 1 -> Positive
        id_to_sentiment = {0: "Negative", 1: "Positive"}
        sentiments = [id_to_sentiment.get(int(idx), f"LABEL_{idx}") for idx in predicted_ids]

        results_df = pd.DataFrame({
            "text": texts,
            "prediction": [int(idx) for idx in predicted_ids],
            "sentiment": sentiments,
            "confidence": [round(float(c), 4) for c in confidences],
        })

        return results_df
