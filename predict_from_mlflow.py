"""
Inference script loading models strictly from the MLflow Model Registry via alias.
Loads 'models:/sentiment-model@champion' (and optionally '@challenger') to evaluate
10 brand-new, unseen sentiment examples without importing Hugging Face pipelines directly.
"""

import os
import argparse
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from src.utils import get_tracking_uri, REGISTERED_MODEL_NAME


def predict_new_data(
    data_path: str = "data/new_data.csv",
    output_path: str = "outputs/new_data_predictions.csv",
    load_challenger: bool = False,
):
    """
    Perform inference on 10 new sentiment examples using the champion model alias.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input data not found at '{data_path}'.")

    new_data = pd.read_csv(data_path)
    print("=================================================================")
    print("     MLflow Inference via Registered Model Aliases")
    print("=================================================================")
    print(f"Tracking URI: {get_tracking_uri()}")
    print(f"Loaded {len(new_data)} unseen test examples from '{data_path}'")

    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Resolve Champion model version dynamically
    champion_mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    champion_run = client.get_run(champion_mv.run_id)
    champion_hf_id = champion_run.data.params.get("hf_model_id", "Unknown")

    print("\n-----------------------------------------------------------------")
    print(f"Loading Champion Model:")
    print(f"  URI:            models:/{REGISTERED_MODEL_NAME}@champion")
    print(f"  Model Version:  {champion_mv.version}")
    print(f"  Run ID:         {champion_mv.run_id}")
    print(f"  HF Model ID:    {champion_hf_id}")
    print("-----------------------------------------------------------------")

    # Load champion strictly through MLflow pyfunc
    champion_pyfunc = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@champion")

    # Predict using Champion
    champion_preds = champion_pyfunc.predict(new_data)

    results_df = pd.DataFrame()
    results_df["id"] = new_data["id"] if "id" in new_data.columns else range(1, len(new_data) + 1)
    results_df["text"] = new_data["text"]
    if "expected_sentiment" in new_data.columns:
        results_df["expected_sentiment"] = new_data["expected_sentiment"]
    results_df["champion_prediction"] = champion_preds["sentiment"]
    results_df["champion_confidence"] = champion_preds["confidence"]

    # Optionally evaluate challenger
    if load_challenger:
        challenger_mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "challenger")
        challenger_run = client.get_run(challenger_mv.run_id)
        challenger_hf_id = challenger_run.data.params.get("hf_model_id", "Unknown")
        print("\nLoading Challenger Model:")
        print(f"  URI:            models:/{REGISTERED_MODEL_NAME}@challenger")
        print(f"  Model Version:  {challenger_mv.version}")
        print(f"  HF Model ID:    {challenger_hf_id}")

        challenger_pyfunc = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@challenger")
        challenger_preds = challenger_pyfunc.predict(new_data)
        results_df["challenger_prediction"] = challenger_preds["sentiment"]
        results_df["challenger_confidence"] = challenger_preds["confidence"]

    # Save outputs
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n[Output] Predictions successfully saved to: {output_path}")

    # Display clean table in terminal
    print("\n=================================================================")
    print("                 NEW DATA PREDICTION RESULTS")
    print("=================================================================")
    for idx, row in results_df.iterrows():
        expected = f" [Expected: {row['expected_sentiment']}]" if "expected_sentiment" in row else ""
        print(f"\nExample {row['id']}: \"{row['text']}\"")
        print(f"  -> Champion Pred: {row['champion_prediction']} (Confidence: {row['champion_confidence']:.4f}){expected}")
        if load_challenger and "challenger_prediction" in row:
            print(f"  -> Challenger Pred: {row['challenger_prediction']} (Confidence: {row['challenger_confidence']:.4f})")

    print("\n=================================================================")
    print("Prediction run complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions with MLflow Champion Model")
    parser.add_argument("--data", default="data/new_data.csv", help="Path to input CSV data")
    parser.add_argument("--output", default="outputs/new_data_predictions.csv", help="Path to output CSV")
    parser.add_argument("--with-challenger", action="store_true", help="Also predict with challenger alias")
    args = parser.parse_args()

    predict_new_data(
        data_path=args.data,
        output_path=args.output,
        load_challenger=args.with_challenger,
    )
