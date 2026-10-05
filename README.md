# ML Evaluation Framework

### Leakage-Safe, Reproducible Machine Learning Model Evaluation

A rigorous machine learning evaluation framework for comparing classification models using **nested cross-validation, leakage-safe preprocessing, hyperparameter tuning, and statistically meaningful evaluation**.

The project focuses on a problem that is often overlooked in applied ML:

> **A model with a high score is not necessarily a model with a trustworthy evaluation.**

This framework is designed to make model comparisons more reliable, reproducible, and resistant to data leakage.

---

## 🎯 Objectives

The framework is built to answer:

* Which model actually generalizes best?
* How much does performance vary across data splits?
* Is hyperparameter tuning introducing optimistic bias?
* Are preprocessing steps leakage-safe?
* Are differences between models meaningful?
* Are predicted probabilities well calibrated?

---

## 🧠 Core Methodology

```text
Raw Dataset
     │
     ▼
Data Audit
     │
     ├── Missing-value analysis
     ├── Data-type validation
     ├── Target analysis
     ├── Class imbalance
     └── Identifier / leakage inspection
     │
     ▼
Leakage-Safe Preprocessing
     │
     ├── Numerical imputation
     ├── Standardization
     └── Categorical encoding
     │
     ▼
Nested Cross-Validation
     │
     ├── Outer CV → unbiased evaluation
     │
     └── Inner CV → hyperparameter tuning
     │
     ▼
Model Comparison
     │
     ├── Logistic Regression
     ├── Random Forest
     ├── HistGradientBoosting
     └── SVM
     │
     ▼
Robust Evaluation
     │
     ├── ROC-AUC
     ├── PR-AUC
     ├── F₂
     ├── MCC
     ├── Log Loss
     ├── Brier Score
     ├── Calibration
     └── Bootstrap confidence intervals
```

---

## 🔬 Dataset

The initial experiment uses the **Telco Customer Churn** classification dataset.

* ~7,000 customer records
* 20+ original features
* Numerical and categorical variables
* Binary target: `Churn`
* Imbalanced target distribution
* `customerID` removed as an identifier
* `TotalCharges` converted from text to numeric
* Missing/invalid numerical values handled inside the preprocessing pipeline

The dataset is used primarily as a benchmark for demonstrating the evaluation methodology.

---

## 🛡️ Leakage-Safe Preprocessing

All preprocessing is encapsulated inside scikit-learn pipelines.

### Numerical features

```text
Missing values
      ↓
Median imputation
      ↓
Standardization
```

### Categorical features

```text
Missing values
      ↓
Most-frequent imputation
      ↓
One-hot encoding
```

Because preprocessing is part of the model pipeline, transformations are fitted only on the training portion of each cross-validation split.

This prevents information from the evaluation fold from leaking into model training.

---

## 🔁 Nested Cross-Validation

The framework uses **nested stratified cross-validation**.

### Outer loop

The outer loop estimates generalization performance on unseen data.

```text
Dataset
   │
   ├── Outer Fold 1 → Test
   ├── Outer Fold 2 → Test
   ├── Outer Fold 3 → Test
   ├── Outer Fold 4 → Test
   └── Outer Fold 5 → Test
```

### Inner loop

Hyperparameters are optimized exclusively within the outer training set.

```text
Outer Training Data
        │
        ▼
   Inner CV
        │
        ├── Hyperparameter search
        ├── Model selection
        └── Best estimator
        │
        ▼
Outer Test Fold
        │
        ▼
Unbiased performance estimate
```

This prevents the test folds from influencing hyperparameter selection.

---

## 🤖 Models

The current framework evaluates four classification algorithms:

| Model                | Purpose                           |
| -------------------- | --------------------------------- |
| Logistic Regression  | Strong linear baseline            |
| Random Forest        | Bagging-based nonlinear model     |
| HistGradientBoosting | Efficient gradient boosting       |
| SVM                  | Margin-based nonlinear classifier |

Each model uses the same evaluation protocol, allowing a fair comparison.

---

## 📊 Nested CV Results

Current ROC-AUC results from the 5-fold outer evaluation:

| Model                | Mean ROC-AUC | Std. Dev. |
| -------------------- | -----------: | --------: |
| HistGradientBoosting |       ~0.846 |         — |
| Logistic Regression  |       ~0.845 |         — |
| Random Forest        |       ~0.841 |         — |
| SVM                  |   ~0.83–0.84 |         — |

*Final aggregate values are stored in `experiments/nested_cv_results.csv`.*

The important point is not simply identifying the highest score, but measuring **performance stability across independent outer folds**.

---

## 📐 Evaluation Metrics

The framework is designed to go beyond ROC-AUC.

### Discrimination

* ROC-AUC
* PR-AUC

### Classification quality

* F₂ Score
* Matthews Correlation Coefficient (MCC)

### Probabilistic quality

* Log Loss
* Brier Score

### Reliability

* Calibration curves
* Bootstrap confidence intervals

This provides a more complete view of model behavior, especially for imbalanced classification problems.

---

## 📈 Why Multiple Metrics?

A single metric can hide important weaknesses.

For example:

```text
ROC-AUC
   │
   └── Ranking ability

PR-AUC
   │
   └── Positive-class performance

F₂
   │
   └── Recall-focused classification

MCC
   │
   └── Balanced correlation-based evaluation

Log Loss
   │
   └── Probability quality

Brier Score
   │
   └── Calibration / probabilistic accuracy
```

The framework therefore treats model evaluation as a **multi-dimensional problem** rather than a single leaderboard score.

---

## 🧪 Reproducibility

Experiments use fixed random seeds and explicit cross-validation strategies.

Example:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

This makes experiments easier to reproduce and compare.

---

## 📁 Project Structure

```text
ml-eval-framework/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   └── 02_model_evaluation.ipynb
│
├── experiments/
│   └── nested_cv_results.csv
│
├── src/
│
├── tests/
│
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

* Python 3.12
* NumPy
* Pandas
* Scikit-learn
* Jupyter Notebook
* Matplotlib
* Git / GitHub

---

## 🚀 Running the Project

Clone the repository:

```bash
git clone https://github.com/roshnishaik78600-cmd/ml-eval-framework.git
cd ml-eval-framework
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Run the notebooks in order:

```text
01_data_audit.ipynb
        ↓
02_model_evaluation.ipynb
```

---

## 🔮 Roadmap

The framework is being extended toward a more complete statistical ML evaluation system.

### Completed

* [x] Dataset audit
* [x] Leakage inspection
* [x] Leakage-safe preprocessing
* [x] Multiple baseline models
* [x] Hyperparameter tuning
* [x] Nested cross-validation
* [x] ROC-AUC comparison

### Next

* [ ] Nested out-of-fold predictions
* [ ] PR-AUC evaluation
* [ ] F₂ and MCC
* [ ] Log Loss and Brier Score
* [ ] Calibration analysis
* [ ] Bootstrap confidence intervals
* [ ] Paired model comparison
* [ ] Automated experiment reports
* [ ] Reusable Python evaluation API
* [ ] Unit tests for evaluation components
* [ ] CLI-based experiment execution

---

## 💡 Key Engineering Principles

This project follows several principles used in production ML experimentation:

**1. Evaluation before optimization**

A sophisticated model is useless if its evaluation protocol is flawed.

**2. Leakage prevention**

Preprocessing and feature transformations belong inside the cross-validation pipeline.

**3. Separation of tuning and evaluation**

Hyperparameter optimization happens inside the inner CV loop; generalization is measured by the outer loop.

**4. Reproducibility**

Experiments use controlled random seeds and explicit evaluation configurations.

**5. Metric diversity**

Model quality is evaluated from discrimination, classification, probability, and calibration perspectives.

**6. Statistical uncertainty**

A small difference in average performance does not automatically mean one model is truly superior.

---

## 📌 Why This Project?

Many ML projects stop at:

```text
Train → Predict → Accuracy → Done
```

This project takes a different approach:

```text
Audit
  ↓
Prevent Leakage
  ↓
Tune Safely
  ↓
Nested Cross-Validation
  ↓
Out-of-Fold Predictions
  ↓
Multiple Metrics
  ↓
Calibration
  ↓
Confidence Intervals
  ↓
Statistical Comparison
  ↓
Trustworthy Model Selection
```

The goal is to build an evaluation system that answers not only:

> **"Which model scored highest?"**

but:

> **"How confident are we that this model actually performs better?"**

---

## 👨‍💻 Author

**Roshni Shaik**

Computer Science Engineering — Cybersecurity
Interests: Machine Learning · AI · Cybersecurity · Reinforcement Learning · ML Systems

---

## ⭐ Project Status

**Active development**

The current implementation establishes the leakage-safe nested cross-validation foundation. Statistical evaluation, calibration, uncertainty estimation, and reusable evaluation components are being added incrementally.
