"""
MLflow Model Registry automation script.
Registers both evaluated models under 'sentiment-model' and dynamically assigns
the 'champion' and 'challenger' aliases based on the experimental decision rule.
"""

import os
import json
import mlflow
from mlflow.tracking import MlflowClient

from src.utils import get_tracking_uri, REGISTERED_MODEL_NAME


def register_and_assign_aliases(summary_file: str = "outputs/metrics_summary.json"):
    """
    Read evaluation summary, register both models to MLflow Model Registry,
    and dynamically assign 'champion' and 'challenger' aliases.
    """
    if not os.path.exists(summary_file):
        raise FileNotFoundError(
            f"Metrics summary file not found at '{summary_file}'. "
            "Please run 'python evaluate.py' first."
        )

    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    run_ids = summary.get("run_ids", {})
    run_id_a = run_ids.get("model_a")
    run_id_b = run_ids.get("model_b")

    if not run_id_a or not run_id_b:
        raise ValueError(f"Run IDs for model_a and model_b not found in {summary_file}.")

    winner = summary.get("winner", "Champion")
    print("=================================================================")
    print(f"MLflow Model Registry Management for: '{REGISTERED_MODEL_NAME}'")
    print(f"Tracking URI: {tracking_uri}")
    print(f"Evaluation Winner: {winner}")
    print("=================================================================")

    # 1. Register Model A (DistilBERT)
    print(f"[Registry] Registering Model A (Run ID: {run_id_a})...")
    model_a_uri = f"runs:/{run_id_a}/model"
    model_version_a = mlflow.register_model(
        model_uri=model_a_uri,
        name=REGISTERED_MODEL_NAME,
    )
    print(f"-> Registered Model A as Version {model_version_a.version}")

    # 2. Register Model B (BERT)
    print(f"[Registry] Registering Model B (Run ID: {run_id_b})...")
    model_b_uri = f"runs:/{run_id_b}/model"
    model_version_b = mlflow.register_model(
        model_uri=model_b_uri,
        name=REGISTERED_MODEL_NAME,
    )
    print(f"-> Registered Model B as Version {model_version_b.version}")

    # 3. Determine Champion and Challenger versions dynamically
    if winner == "Challenger":
        champion_version = model_version_b.version
        champion_model_id = summary["models"]["model_b_challenger"]["hf_model_id"]
        challenger_version = model_version_a.version
        challenger_model_id = summary["models"]["model_a_champion"]["hf_model_id"]
    else:
        # Default Champion retained
        champion_version = model_version_a.version
        champion_model_id = summary["models"]["model_a_champion"]["hf_model_id"]
        challenger_version = model_version_b.version
        challenger_model_id = summary["models"]["model_b_challenger"]["hf_model_id"]

    print("\n-----------------------------------------------------------------")
    print(f"Assigning Registry Aliases:")
    print(f"  - 'champion'   -> Version {champion_version} ({champion_model_id})")
    print(f"  - 'challenger' -> Version {challenger_version} ({challenger_model_id})")
    print("-----------------------------------------------------------------")

    # Set aliases in MLflow Model Registry
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=str(champion_version),
    )
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="challenger",
        version=str(challenger_version),
    )

    # Verification of assigned aliases
    print("\n[Verification] Fetching current aliases from Registry:")
    reg_model = client.get_registered_model(REGISTERED_MODEL_NAME)
    for alias_name, ver in reg_model.aliases.items():
        print(f"  Alias '@{alias_name}' points to Version: {ver}")

    print("\nModel registration and alias assignment complete!")


if __name__ == "__main__":
    register_and_assign_aliases()
