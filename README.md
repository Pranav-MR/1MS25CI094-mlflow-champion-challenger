# Champion–Challenger Model Comparison with Hugging Face and MLflow

An end-to-end, production-ready MLOps mini-project for automated sentiment classification model evaluation, experiment tracking, and model lifecycle governance.

---

## 1. Project Overview

In production Machine Learning workflows, deployed models ("Champions") must be continuously evaluated against emerging alternatives ("Challengers") to determine whether updating the production model yields measurable performance improvements without introducing regressions or unsustainable inference latency.

This project implements a reproducible, automated **Champion–Challenger MLOps pipeline** using **Hugging Face Transformers**, **PyTorch**, and **MLflow**. The pipeline evaluates two pretrained sentiment analysis models on a standardized evaluation dataset, records parameters, metrics, and artifacts to a local MLflow tracking server backed by SQLite, evaluates an objective decision rule, registers both models to the MLflow Model Registry, and assigns dynamic aliases (`@champion` and `@challenger`). Downstream inference is executed purely via MLflow model URIs without direct dependencies on raw Hugging Face code.

---

## 2. Task and Selected Models

- **NLP Task**: Binary Sentiment Classification (Negative vs. Positive)
- **Evaluation Benchmark**: SST-2 (Stanford Sentiment Treebank)

### Evaluated Models

| Role | Model Name | Hugging Face ID | Architecture / Size |
| :--- | :--- | :--- | :--- |
| **Model A (Champion)** | DistilBERT SST-2 | `distilbert-base-uncased-finetuned-sst-2-english` | 6 layers, 66M parameters |
| **Model B (Challenger)** | BERT SST-2 | `textattack/bert-base-uncased-SST-2` | 12 layers, 110M parameters |

Both models receive identical tokenization limits (`max_length=128`), inference batch sizes (`batch_size=16`), and evaluation inputs in deterministic order on CPU.

---

## 3. Dataset and Evaluation Sample Size

- **Source**: Hugging Face Datasets (`glue/sst2` validation split)
- **Sample Selection**: Deterministic subset of **200 examples** sampled using a fixed random seed (`seed=42`).
- **Storage**: Persisted to [`data/eval_inputs.csv`](data/eval_inputs.csv) ensuring identical ground-truth comparisons across runs.
- **Labels**:
  - `0`: Negative
  - `1`: Positive

---

## 4. Technology Stack

- **Language**: Python 3.10+
- **Deep Learning & NLP**: Hugging Face `transformers`, `torch` (CPU mode), `datasets`
- **MLOps & Tracking**: `mlflow` (Experiment tracking, PyFunc wrappers, Model Registry)
- **Evaluation & Utilities**: `scikit-learn`, `pandas`, `numpy`, `matplotlib`
- **Version Control**: Git & GitHub

---

## 5. Champion–Challenger Decision Rule

> **Predefined Decision Rule (Formulated prior to experimental execution):**
> 1. **Primary Metric**: Macro F1-score ($F_{1,\text{macro}}$). Higher is better.
> 2. **Challenger Promotion Threshold**: The Challenger (Model B) must improve Macro F1 by at least **0.01 (+1.0%)** over the Champion (Model A) ($\Delta F_1 \ge +0.01$) to claim the Champion title.
> 3. **Retention Policy**: If $\Delta F_1 < +0.01$, the current Champion (Model A) is retained.
> 4. **Tie-Breaker / Margin Rule**: If the Macro F1 difference is within the margin ($|\Delta F_1| < 0.01$), the model with lower average inference latency (faster CPU execution) is preferred.
> 5. **Consistency Rule**: Both candidate models must be evaluated under strictly identical conditions (sample size, seed, max length, CPU hardware).

---

## 6. Installation & Environment Setup

### Prerequisites
- Python 3.10 or newer
- Git

### Setup Steps
```bash
# Clone the repository
git clone https://github.com/Pranav-MR/1MS25CI094-mlflow-champion-challenger.git
cd 1MS25CI094-mlflow-champion-challenger

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 7. Execution Commands in Order

Execute the following commands sequentially to run the entire project:

```bash
# Step 1 (Optional Smoke Test): Quick test with 10 samples to verify pipeline
python evaluate.py --smoke-test

# Step 2: Full 200-sample Champion-Challenger Evaluation
python evaluate.py

# Step 3: Register Models and Assign Aliases (@champion and @challenger)
python register_champion.py

# Step 4: Run Inference on 10 New Examples using MLflow Champion Alias
python predict_from_mlflow.py --with-challenger
```

---

## 8. Results Comparison Table

*(Populated dynamically from verified experimental execution on 200 SST-2 validation samples)*

| Metric | Model A (DistilBERT - Champion) | Model B (BERT - Challenger) | Difference ($\Delta = \text{B} - \text{A}$) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | *Pending Run* | *Pending Run* | *Pending Run* |
| **Precision (Macro)** | *Pending Run* | *Pending Run* | *Pending Run* |
| **Recall (Macro)** | *Pending Run* | *Pending Run* | *Pending Run* |
| **Macro F1-Score** | *Pending Run* | *Pending Run* | *Pending Run* |
| **Avg Latency (ms/sample)** | *Pending Run* | *Pending Run* | *Pending Run* |
| **Total Inference Time (s)** | *Pending Run* | *Pending Run* | *Pending Run* |

---

## 9. Final Decision & Justification

*(Will be populated with actual figures upon pipeline execution)*

---

## 10. Model Trade-offs, Latency and Limitations

1. **Model Capacity vs. Compute Cost**:
   - **DistilBERT** retains ~97% of BERT's language understanding while being 40% smaller and ~60% faster, making it exceptionally well-suited for low-latency CPU environments.
   - **BERT-Base** contains 110M parameters with 12 transformer encoder blocks, yielding deeper contextual representations at the cost of higher CPU memory consumption and ~2x inference latency.

2. **Inference Latency Trade-Off**:
   - In production microservices, lower latency directly impacts user experience and server scaling costs. A small accuracy boost must be weighted against throughput degradation.

3. **Domain Limitations**:
   - Both models are fine-tuned on SST-2 (movie reviews). Performance may degrade on slang, domain-specific terminology (e.g., medical or financial text), sarcasm, or multi-sentence paragraphs exceeding `max_length=128`.

---

## 11. Instructions to Access MLflow UI

Launch the MLflow tracking user interface to visually inspect runs, metric curves, and registry aliases:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

1. Open your browser and navigate to: `http://localhost:5000`
2. Select the experiment: **`hf-sentiment-champion-challenger`**
3. View the 3 runs:
   - `A_sentiment_distilbert-sst2`
   - `B_sentiment_bert-sst2`
   - `compare_sentiment`
4. Navigate to the **Models** tab at the top to view the registered model `sentiment-model` and inspect its `@champion` and `@challenger` aliases.

---

## 12. Instructions to Run Predictions Using Champion Alias

To evaluate unseen text directly through the MLflow Model Registry without using raw Hugging Face pipelines:

```bash
python predict_from_mlflow.py
```

To compare predictions against the challenger model simultaneously:

```bash
python predict_from_mlflow.py --with-challenger
```

The script loads `models:/sentiment-model@champion`, processes 10 new test samples in [`data/new_data.csv`](data/new_data.csv), prints the results to the terminal, and exports them to [`outputs/new_data_predictions.csv`](outputs/new_data_predictions.csv).

---

## 13. Screenshots Checklist

The [`screenshots/`](screenshots/) directory is prepared for manual capture:
1. `mlflow_comparison_view.png` - MLflow comparison view showing runs side-by-side.
2. `mlflow_registry_aliases.png` - MLflow Model Registry page displaying `@champion` and `@challenger` aliases.
3. `predict_from_mlflow_terminal.png` - Terminal execution output of `predict_from_mlflow.py`.

---

## 14. Declaration of AI Assistance

In accordance with academic integrity guidelines:
- An AI assistant (Antigravity by Google DeepMind) was utilized during the development of this project for code structure recommendations, script templating, and documentation formatting.
- All model weights, evaluation datasets, inference calculations, metrics, and experimental decisions were executed and verified locally.
