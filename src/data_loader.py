"""
Data loader module for SST-2 validation dataset.
Loads a deterministic evaluation subset (200 samples) and persists it to data/eval_inputs.csv.
"""

import os
import pandas as pd
from datasets import load_dataset


def load_evaluation_data(
    num_samples: int = 200,
    seed: int = 42,
    output_path: str = "data/eval_inputs.csv",
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Load the Hugging Face SST-2 validation dataset.
    Select a deterministic subset of `num_samples` examples using a fixed random seed.
    Save the evaluation data to `output_path`.

    Parameters:
    -----------
    num_samples: Number of evaluation samples (default: 200)
    seed: Random seed for deterministic selection (default: 42)
    output_path: Destination CSV filepath
    force_download: If True, re-downloads and re-samples even if CSV exists.

    Returns:
    --------
    pd.DataFrame containing columns: [id, text, label, label_text]
    """
    if not force_download and os.path.exists(output_path):
        print(f"[DataLoader] Found existing evaluation data at {output_path}. Loading...")
        df = pd.read_csv(output_path)
        if len(df) == num_samples and "text" in df.columns and "label" in df.columns:
            return df
        print(f"[DataLoader] Existing data has {len(df)} rows, expected {num_samples}. Re-fetching...")

    print(f"[DataLoader] Fetching SST-2 validation split from Hugging Face Datasets...")
    dataset = None
    errors = []

    # Try standard Glue benchmark identifier first, then stanfordnlp/sst2
    for ds_name in [("glue", "sst2"), ("stanfordnlp/sst2", None)]:
        try:
            if ds_name[1]:
                dataset = load_dataset(ds_name[0], ds_name[1], split="validation")
            else:
                dataset = load_dataset(ds_name[0], split="validation")
            print(f"[DataLoader] Successfully loaded dataset: {ds_name}")
            break
        except Exception as e:
            errors.append(f"{ds_name}: {str(e)}")

    if dataset is None:
        raise RuntimeError(
            f"Failed to download SST-2 validation dataset. Errors encountered:\n"
            + "\n".join(errors)
        )

    full_df = dataset.to_pandas()

    # Identify sentence column (either 'sentence' or 'text')
    text_col = "sentence" if "sentence" in full_df.columns else "text"
    if text_col not in full_df.columns or "label" not in full_df.columns:
        raise ValueError(f"Unexpected dataset schema. Available columns: {list(full_df.columns)}")

    # Deterministic sampling with fixed random seed
    if len(full_df) < num_samples:
        raise ValueError(f"Dataset has only {len(full_df)} samples, requested {num_samples}.")

    sampled_df = full_df.sample(n=num_samples, random_state=seed).reset_index(drop=True)

    # Standardize column names
    clean_df = pd.DataFrame()
    clean_df["id"] = sampled_df.index
    clean_df["text"] = sampled_df[text_col]
    clean_df["label"] = sampled_df["label"].astype(int)
    clean_df["label_text"] = clean_df["label"].map({0: "Negative", 1: "Positive"})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clean_df.to_csv(output_path, index=False)
    print(f"[DataLoader] Saved {len(clean_df)} deterministic evaluation samples to {output_path}")

    return clean_df


if __name__ == "__main__":
    df = load_evaluation_data(num_samples=200, seed=42)
    print("Evaluation dataset sample:")
    print(df.head())
    print("\nClass distribution:")
    print(df["label_text"].value_counts())
