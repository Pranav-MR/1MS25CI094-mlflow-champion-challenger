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
| **Model A (Initial Champion)** | DistilBERT SST-2 | `distilbert-base-uncased-finetuned-sst-2-english` | 6 layers, 66M parameters |
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
- **MLOps & Tracking**: `mlflow` (Experiment tracking, PyFunc custom model wrappers, Model Registry)
- **Evaluation & Utilities**: `scikit-learn`, `pandas`, `numpy`, `matplotlib`
- **Version Control**: Git & GitHub

---

## 5. Champion–Challenger Decision Rule

> **Predefined Decision Rule (Formulated prior to experimental execution):**
> 1. **Primary Metric**: Macro F1-score ($F_{1,\text{macro}}$). Higher is better.
> 2. **Challenger Promotion Threshold**: The Challenger (Model B) must improve Macro F1 by at least **0.01 (+1.0%)** over the Champion (Model A) ($\Delta F_1 \ge +0.01$) to replace the Champion.
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

## 7. Exact Execution Commands in Order

Execute the following commands sequentially to run the entire project:

```bash
# Step 1 (Optional Smoke Test): Quick test with 10 samples to verify pipeline
python evaluate.py --smoke-test

# Step 2: Full 200-sample Champion-Challenger Evaluation
python evaluate.py

# Step 3: Register Models and Assign Dynamic Aliases (@champion and @challenger)
python register_champion.py

# Step 4: Run Inference on 10 New Examples using MLflow Champion Alias
python predict_from_mlflow.py --with-challenger
```

---

## 8. Results Comparison Table

*(Populated from verified experimental execution on 200 deterministic SST-2 validation samples)*

| Metric | Model A (DistilBERT - Initial Champion) | Model B (BERT - Challenger) | Difference ($\Delta = \text{B} - \text{A}$) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.9350 (93.50%) | **0.9450 (94.50%)** | +0.0100 (+1.00%) |
| **Precision (Macro)** | 0.9359 | **0.9442** | +0.0083 |
| **Recall (Macro)** | 0.9331 | **0.9450** | +0.0119 |
| **Macro F1-Score (Primary)** | 0.9343 | **0.9446** | **+0.0103 (+1.03%)** |
| **Avg Latency (ms/sample)** | **10.90 ms** | 21.68 ms | +10.78 ms (+98.9%) |
| **Total Inference Time (s)** | **2.18 s** | 4.34 s | +2.16 s |

---

## 9. Final Decision & Justification

**Winner**: **Model B (Challenger: `textattack/bert-base-uncased-SST-2`)**  
**Assigned Registry Alias**: `@champion` (Version 2)  
**Incumbent Reassignment**: `@challenger` (Version 1)

### Formal Justification (205 words)
Based on the predefined champion-challenger decision rule established prior to testing, Model B (Challenger: `textattack/bert-base-uncased-SST-2`) has officially replaced Model A (Champion: `distilbert-base-uncased-finetuned-sst-2-english`) as the new production Champion.

The primary evaluation criterion stipulated that the Challenger must improve the Macro F1-score by at least 0.01 (+1.00%) over the incumbent model on the identical 200-sample SST-2 validation dataset. Model B achieved a Macro F1-score of 0.9446 compared to Model A's 0.9343, generating a net performance delta of $\Delta F_1 = +0.0103$. Because $+0.0103 \ge 0.01$, the Challenger successfully met the mandatory threshold for promotion. Furthermore, Model B demonstrated superior balanced recall (0.9450 vs. 0.9331) and higher overall classification accuracy (94.50% vs. 93.50%).

Although Model B's average inference latency is roughly twice that of Model A (21.68 ms vs. 10.90 ms on CPU) due to its 12-layer 110M parameter architecture, the decision rule explicitly requires latency tie-breaking only when the primary metric difference falls strictly within the $\pm 0.01$ margin. Because the performance improvement exceeded the promotion threshold, Model B is crowned the new Champion and registered under the MLflow Model Registry alias `@champion` (Version 2), while Model A has been reassigned to `@challenger` (Version 1).

---

## 10. Model Trade-offs, Latency and Limitations

1. **Model Capacity vs. Compute Cost**:
   - **DistilBERT** retains ~97% of BERT's language understanding capability while being 40% smaller (66M vs. 110M parameters) and ~50% faster (10.90 ms vs. 21.68 ms per inference on CPU).
   - **BERT-Base** leverages 12 full transformer encoder blocks with 12 self-attention heads, yielding superior contextual representations on syntactically complex or sarcastic phrasing, but requiring higher memory throughput and CPU computation time.

2. **Inference Latency Trade-Off**:
   - In low-latency interactive applications, DistilBERT's 10.90 ms latency delivers nearly double the request throughput. However, in batch scoring or quality-critical sentiment pipelines where accuracy is paramount, BERT's +1.03% Macro F1 gain justifies the 10.78 ms additional latency.

3. **Disagreement Analysis Insights**:
   - Evaluation on 200 samples revealed 8 concrete disagreements ([`outputs/disagreement_examples.csv`](outputs/disagreement_examples.csv)). BERT excelled at detecting nuanced negative reviews containing complex sarcasm or vocabulary (e.g., *"a bloated gasbag thesis grotesquely impressed by its own gargantuan aura of self-importance"*, where DistilBERT mispredicted Positive with 84.9% confidence, while BERT correctly predicted Negative with 99.7% confidence).

4. **Domain Limitations**:
   - Both models are fine-tuned on SST-2 (movie review snippets). Accuracy may degrade on cross-domain texts (such as financial filings or clinical notes), multi-sentence documents exceeding `max_length=128`, or domain-specific jargon.

---

## 11. Instructions to Access MLflow UI

Launch the MLflow tracking user interface to visually inspect runs, parameter comparisons, metric charts, and model registry aliases:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

1. Open your browser and navigate to: `http://localhost:5000`
2. Select the experiment: **`hf-sentiment-champion-challenger`**
3. View the 3 runs:
   - `A_sentiment_distilbert-sst2`
   - `B_sentiment_bert-sst2`
   - `compare_sentiment`
4. Inspect logged parameters (`max_length`, `batch_size`, `seed`, `device`), metrics (`accuracy`, `macro_f1`, `avg_latency_ms`), tags (`student_usn`), and artifacts (`predictions_*.csv`, `confusion_matrix_*.png`, `model_comparison.png`).
5. Navigate to the **Models** tab at the top to view the registered model `sentiment-model` and verify the assigned `@champion` (Version 2) and `@challenger` (Version 1) aliases.

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

The script dynamically resolves `models:/sentiment-model@champion`, processes the 10 unseen test samples in [`data/new_data.csv`](data/new_data.csv), displays the predictions and confidence scores in the terminal, and exports them to [`outputs/new_data_predictions.csv`](outputs/new_data_predictions.csv).

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
