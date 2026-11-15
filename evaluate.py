"""
Main entry point for Champion-Challenger Model Evaluation Pipeline.
Evaluates Model A (DistilBERT) and Model B (BERT) on SST-2 validation dataset,
logs experiments to MLflow, and executes the predefined decision rule.
"""

import argparse
import sys
import os

from src.data_loader import load_evaluation_data
from src.evaluate import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run Champion-Challenger Evaluation Pipeline with Hugging Face and MLflow"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick smoke test on 10 examples instead of full 200 samples",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of evaluation samples to draw from SST-2 (default: 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size (default: 16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sample selection (default: 42)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device (default: cpu)",
    )

    args = parser.parse_args()

    num_samples = 10 if args.smoke_test else args.num_samples
    eval_csv_path = "data/smoke_inputs.csv" if args.smoke_test else "data/eval_inputs.csv"

    print("=================================================================")
    print("  MLOps Champion-Challenger Model Comparison: Hugging Face & MLflow")
    print("=================================================================")
    print(f"Mode: {'Smoke Test (10 samples)' if args.smoke_test else f'Full Evaluation ({num_samples} samples)'}")
    print(f"Dataset path: {eval_csv_path}")
    print(f"Device: {args.device} | Batch Size: {args.batch_size} | Seed: {args.seed}")
    print("-----------------------------------------------------------------")

    # Step 1: Load evaluation dataset
    eval_df = load_evaluation_data(
        num_samples=num_samples,
        seed=args.seed,
        output_path=eval_csv_path,
    )
    print(f"Loaded {len(eval_df)} evaluation examples successfully.")

    # Step 2: Run pipeline across 3 MLflow runs
    summary = run_pipeline(
        eval_df=eval_df,
        batch_size=args.batch_size,
        max_length=128,
        seed=args.seed,
        device=args.device,
        log_models_to_mlflow=True,
    )

    print("\n=================================================================")
    print("                     EVALUATION SUMMARY")
    print("=================================================================")
    print(f"Winner: {summary['winner']} ({summary['winner_model_id']})")
    print(f"Decision Reason: {summary['decision_reason']}")
    print(f"Delta Macro F1 (Challenger - Champion): {summary['delta_primary_metric']}")
    print("-----------------------------------------------------------------")
    print("Model A (Champion) Metrics:")
    for k, v in summary["models"]["model_a_champion"]["metrics"].items():
        print(f"  - {k}: {v}")
    print("Model B (Challenger) Metrics:")
    for k, v in summary["models"]["model_b_challenger"]["metrics"].items():
        print(f"  - {k}: {v}")
    print("=================================================================")
    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
