"""
Core evaluation and comparison engine for Champion-Challenger pipeline.
Conducts batch evaluation on SST-2 data, calculates metrics, tracks experiments in MLflow,
and executes the decision rule.
"""

import os
import time
import shutil
import tempfile
import numpy as np
import pandas as pd
import torch
import transformers
import mlflow
import mlflow.pyfunc

from src.model_loader import MODEL_CONFIGS, load_tokenizer_and_model
from src.mlflow_wrapper import HuggingFaceSentimentWrapper
from src.utils import (
    get_student_usn,
    get_tracking_uri,
    EXPERIMENT_NAME,
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_model_comparison,
    save_metrics_summary,
)


def evaluate_model_on_data(
    model_key: str,
    eval_df: pd.DataFrame,
    batch_size: int = 16,
    max_length: int = 128,
    device: str = "cpu",
):
    """
    Evaluate a specific model on the provided evaluation dataset.

    Returns:
    - metrics: dictionary of accuracy, precision, recall, macro_f1, avg_latency_ms, total_inference_s
    - predictions_df: dataframe with individual predictions, confidence, and correctness
    - tokenizer, model: loaded Hugging Face objects
    """
    cfg = MODEL_CONFIGS[model_key]
    hf_model_id = cfg["hf_model_id"]

    tokenizer, model = load_tokenizer_and_model(hf_model_id, device=device)

    texts = eval_df["text"].tolist()
    true_labels = eval_df["label"].tolist()

    all_preds = []
    all_confidences = []
    batch_latencies_ms = []

    start_total_time = time.perf_counter()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        t0 = time.perf_counter()

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).cpu().numpy()
            confs = torch.max(probs, dim=-1).values.cpu().numpy()

        t1 = time.perf_counter()
        batch_latency = (t1 - t0) * 1000.0  # ms for this batch
        per_item_latency = batch_latency / len(batch_texts)

        for _ in batch_texts:
            batch_latencies_ms.append(per_item_latency)

        all_preds.extend(preds.tolist())
        all_confidences.extend(confs.tolist())

    total_inference_time = time.perf_counter() - start_total_time

    metrics = compute_classification_metrics(
        y_true=true_labels,
        y_pred=all_preds,
        latency_ms_list=batch_latencies_ms,
        total_time_s=total_inference_time,
    )

    pred_text_map = {0: "Negative", 1: "Positive"}
    preds_df = pd.DataFrame({
        "id": eval_df["id"].tolist() if "id" in eval_df.columns else list(range(len(eval_df))),
        "text": texts,
        "true_label": true_labels,
        "true_label_text": [pred_text_map[t] for t in true_labels],
        "predicted_label": all_preds,
        "predicted_label_text": [pred_text_map[p] for p in all_preds],
        "confidence": [round(float(c), 4) for c in all_confidences],
        "correct": [int(t == p) for t, p in zip(true_labels, all_preds)],
    })

    return metrics, preds_df, tokenizer, model


def run_pipeline(
    eval_df: pd.DataFrame,
    batch_size: int = 16,
    max_length: int = 128,
    seed: int = 42,
    device: str = "cpu",
    log_models_to_mlflow: bool = True,
):
    """
    Execute full Champion-Challenger pipeline across 3 MLflow runs:
    1. A_sentiment_distilbert-sst2
    2. B_sentiment_bert-sst2
    3. compare_sentiment
    """
    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    print(f"[MLflow] Tracking URI set to: {tracking_uri}")

    mlflow.set_experiment(EXPERIMENT_NAME)
    student_usn = get_student_usn()

    standard_tags = {
        "student_usn": student_usn,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "mlflow_version": mlflow.__version__,
        "dataset_name": "glue/sst2",
    }

    results = {}
    run_ids = {}

    # --- 1. Evaluate Model A (Champion) and Model B (Challenger) ---
    for model_key in ["model_a", "model_b"]:
        cfg = MODEL_CONFIGS[model_key]
        run_name = cfg["run_name"]
        hf_model_id = cfg["hf_model_id"]
        role = cfg["role"]

        print(f"\n==========================================")
        print(f"Evaluating {cfg['display_name']} [{hf_model_id}]")
        print(f"==========================================")

        metrics, preds_df, tokenizer, model = evaluate_model_on_data(
            model_key=model_key,
            eval_df=eval_df,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )

        # Save predictions CSV
        preds_path = cfg["output_predictions_file"]
        os.makedirs(os.path.dirname(preds_path), exist_ok=True)
        preds_df.to_csv(preds_path, index=False)
        print(f"[Output] Predictions saved to {preds_path}")

        # Save confusion matrix plot
        cm_path = cfg["confusion_matrix_file"]
        plot_confusion_matrix(
            y_true=preds_df["true_label"],
            y_pred=preds_df["predicted_label"],
            class_names=["Negative", "Positive"],
            output_path=cm_path,
            title=f"Confusion Matrix: {cfg['display_name']}",
        )
        print(f"[Output] Confusion matrix saved to {cm_path}")

        results[model_key] = {
            "metrics": metrics,
            "preds_df": preds_df,
            "cfg": cfg,
        }

        # Start MLflow run for this model
        with mlflow.start_run(run_name=run_name) as run:
            run_ids[model_key] = run.info.run_id
            print(f"[MLflow] Started run '{run_name}' (Run ID: {run.info.run_id})")

            # Log parameters
            mlflow.log_params({
                "hf_model_id": hf_model_id,
                "task": "Sentiment Classification",
                "role": role,
                "max_length": max_length,
                "batch_size": batch_size,
                "eval_samples": len(eval_df),
                "seed": seed,
                "device": device,
            })

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log tags
            mlflow.set_tags(standard_tags)

            # Log artifacts: predictions CSV and confusion matrix image
            mlflow.log_artifact(preds_path, artifact_path="evaluation")
            mlflow.log_artifact(cm_path, artifact_path="evaluation")

            # Package and log pyfunc model if requested
            if log_models_to_mlflow:
                print(f"[MLflow] Packaging pyfunc model for {role}...")
                with tempfile.TemporaryDirectory() as temp_dir:
                    model_save_dir = os.path.join(temp_dir, "hf_model")
                    tokenizer.save_pretrained(model_save_dir)
                    model.save_pretrained(model_save_dir)

                    artifacts = {"model_dir": model_save_dir}
                    pyfunc_model = HuggingFaceSentimentWrapper(
                        hf_model_id=hf_model_id,
                        max_length=max_length,
                    )

                    mlflow.pyfunc.log_model(
                        artifact_path="model",
                        python_model=pyfunc_model,
                        artifacts=artifacts,
                    )
                print(f"[MLflow] Successfully logged pyfunc model artifact for {role}.")

    # --- 2. Compare Models and Execute Predefined Decision Rule ---
    print("\n==========================================")
    print("Executing Champion-Challenger Decision Rule")
    print("==========================================")

    f1_a = results["model_a"]["metrics"]["macro_f1"]
    f1_b = results["model_b"]["metrics"]["macro_f1"]
    lat_a = results["model_a"]["metrics"]["avg_latency_ms"]
    lat_b = results["model_b"]["metrics"]["avg_latency_ms"]

    delta_f1 = round(f1_b - f1_a, 4)
    margin = 0.01

    # Predefined Decision Rule:
    # 1. Primary metric: Macro F1-score (higher is better).
    # 2. Challenger must improve Macro F1 by at least 0.01 to replace Champion.
    # 3. If improvement is < 0.01, retain Champion.
    # 4. If difference is within margin, prefer faster model.
    if delta_f1 >= margin:
        winner = "Challenger"
        winner_key = "model_b"
        loser_key = "model_a"
        reason = f"Challenger outperformed Champion by {delta_f1:.4f} (>= margin of {margin})."
    elif delta_f1 <= -margin:
        winner = "Champion"
        winner_key = "model_a"
        loser_key = "model_b"
        reason = f"Champion outperformed Challenger by {-delta_f1:.4f} (>= margin of {margin})."
    else:
        # Within margin [-0.01, 0.01] -> prefer faster model (lower latency)
        if lat_b < lat_a:
            winner = "Challenger"
            winner_key = "model_b"
            loser_key = "model_a"
            reason = f"F1 difference ({delta_f1:.4f}) is within margin ({margin}); Challenger is faster ({lat_b:.2f}ms vs {lat_a:.2f}ms)."
        else:
            winner = "Champion"
            winner_key = "model_a"
            loser_key = "model_b"
            reason = f"F1 difference ({delta_f1:.4f}) is within margin ({margin}); Champion is retained (latency: {lat_a:.2f}ms vs {lat_b:.2f}ms)."

    print(f"Model A (Champion) Macro F1: {f1_a:.4f} | Latency: {lat_a:.2f} ms")
    print(f"Model B (Challenger) Macro F1: {f1_b:.4f} | Latency: {lat_b:.2f} ms")
    print(f"Delta Macro F1 (B - A): {delta_f1:.4f} (Threshold: {margin})")
    print(f"Decision Winner: {winner}")
    print(f"Justification: {reason}")

    # Generate disagreement analysis (find where predictions differ)
    preds_a = results["model_a"]["preds_df"]
    preds_b = results["model_b"]["preds_df"]

    disagreements = []
    for idx in range(len(preds_a)):
        row_a = preds_a.iloc[idx]
        row_b = preds_b.iloc[idx]
        if row_a["predicted_label"] != row_b["predicted_label"]:
            true_label_text = row_a["true_label_text"]
            pred_a_text = row_a["predicted_label_text"]
            pred_b_text = row_b["predicted_label_text"]
            text = row_a["text"]

            # Concise explanation of the disagreement
            if row_a["correct"] and not row_b["correct"]:
                explanation = f"Champion correctly identified as {pred_a_text}; Challenger misclassified as {pred_b_text}."
            elif not row_a["correct"] and row_b["correct"]:
                explanation = f"Challenger correctly identified as {pred_b_text}; Champion misclassified as {pred_a_text}."
            else:
                explanation = f"Both models deviated or interpreted nuanced sentiment differently."

            disagreements.append({
                "sample_id": row_a["id"],
                "text": text,
                "ground_truth": true_label_text,
                "distilbert_prediction": pred_a_text,
                "distilbert_confidence": row_a["confidence"],
                "bert_prediction": pred_b_text,
                "bert_confidence": row_b["confidence"],
                "explanation": explanation,
            })

    disagreements_df = pd.DataFrame(disagreements)
    disagreements_path = "outputs/disagreement_examples.csv"
    os.makedirs(os.path.dirname(disagreements_path), exist_ok=True)
    disagreements_df.to_csv(disagreements_path, index=False)
    print(f"[Output] Identified {len(disagreements_df)} disagreements. Saved to {disagreements_path}")

    # Generate comparison plot
    comparison_plot_path = "outputs/model_comparison.png"
    plot_model_comparison(
        metrics_a=results["model_a"]["metrics"],
        metrics_b=results["model_b"]["metrics"],
        output_path=comparison_plot_path,
    )
    print(f"[Output] Comparison plot saved to {comparison_plot_path}")

    # Save metrics summary JSON
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "glue/sst2",
        "sample_size": len(eval_df),
        "primary_metric": "macro_f1",
        "margin": margin,
        "delta_primary_metric": delta_f1,
        "winner": winner,
        "winner_model_id": MODEL_CONFIGS[winner_key]["hf_model_id"],
        "decision_reason": reason,
        "run_ids": run_ids,
        "models": {
            "model_a_champion": {
                "hf_model_id": MODEL_CONFIGS["model_a"]["hf_model_id"],
                "role": "Champion",
                "run_id": run_ids.get("model_a"),
                "metrics": results["model_a"]["metrics"],
            },
            "model_b_challenger": {
                "hf_model_id": MODEL_CONFIGS["model_b"]["hf_model_id"],
                "role": "Challenger",
                "run_id": run_ids.get("model_b"),
                "metrics": results["model_b"]["metrics"],
            },
        },
        "disagreements_count": len(disagreements_df),
    }

    summary_path = "outputs/metrics_summary.json"
    save_metrics_summary(summary_data, summary_path)
    print(f"[Output] Metrics summary saved to {summary_path}")

    # --- 3. MLflow compare_sentiment Run ---
    with mlflow.start_run(run_name="compare_sentiment") as comp_run:
        run_ids["compare"] = comp_run.info.run_id
        print(f"[MLflow] Started run 'compare_sentiment' (Run ID: {comp_run.info.run_id})")

        mlflow.log_params({
            "primary_metric": "macro_f1",
            "margin": margin,
            "winner": winner,
            "winner_model_id": MODEL_CONFIGS[winner_key]["hf_model_id"],
            "model_a_run_id": run_ids["model_a"],
            "model_b_run_id": run_ids["model_b"],
        })

        mlflow.log_metrics({
            "delta_primary_metric": delta_f1,
            "model_a_macro_f1": f1_a,
            "model_b_macro_f1": f1_b,
            "model_a_accuracy": results["model_a"]["metrics"]["accuracy"],
            "model_b_accuracy": results["model_b"]["metrics"]["accuracy"],
            "model_a_avg_latency_ms": lat_a,
            "model_b_avg_latency_ms": lat_b,
            "disagreements_count": float(len(disagreements_df)),
        })

        mlflow.set_tags(standard_tags)

        # Log comparison artifacts
        mlflow.log_artifact(comparison_plot_path, artifact_path="comparison")
        mlflow.log_artifact(disagreements_path, artifact_path="comparison")
        mlflow.log_artifact(summary_path, artifact_path="comparison")

    return summary_data
