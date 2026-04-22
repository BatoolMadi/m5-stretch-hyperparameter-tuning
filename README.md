# Module 5 Week B — Stretch: Hyperparameter Tuning & Nested Cross-Validation

## Overview
This project explores systematic hyperparameter tuning using GridSearchCV and demonstrates the importance of unbiased model evaluation through nested cross-validation.

The goal is to understand how hyperparameter selection can introduce optimistic bias and how nested cross-validation provides a more reliable performance estimate.

---

## Dataset
Telecom churn dataset containing customer information and a binary target variable (`churned`).

---

## Part 1 — GridSearchCV
- Tuned a Random Forest model using GridSearchCV
- Used 5-fold stratified cross-validation with F1 score
- Explored the following hyperparameters:
  - n_estimators: [50, 100, 200]
  - max_depth: [3, 5, 10, 20, None]
  - min_samples_split: [2, 5, 10]
- Visualized results using a heatmap

📊 Output:
- Best hyperparameters
- Best cross-validation F1 score
- Heatmap (`heatmap.png`)

---

## Part 2 — Nested Cross-Validation
- Implemented nested cross-validation with:
  - Outer loop: 5-fold StratifiedKFold
  - Inner loop: GridSearchCV
- Compared two models:
  - Random Forest (with class_weight='balanced')
  - Decision Tree (with class_weight='balanced')

📊 Results:

| Model          | Inner Score | Outer Score | Gap |
|----------------|------------|------------|------|
| Random Forest  | 0.493      | 0.491      | 0.002 |
| Decision Tree  | 0.476      | 0.468      | 0.008 |

---

## Key Insight
GridSearchCV.best_score_ can be optimistically biased because it evaluates performance on the same data used for hyperparameter tuning. Nested cross-validation addresses this by separating model selection and evaluation, similar to using a held-out test set.

---

## Files
- `grid_search_rf.py` → Part 1 implementation
- `nested_cv.py` → Part 2 implementation
- `analysis.md` → Written analysis
- `heatmap.png` → Visualization output
- `requirements.txt` → Dependencies

---

## How to Run

```bash
pip install -r requirements.txt
python grid_search_rf.py
python nested_cv.py