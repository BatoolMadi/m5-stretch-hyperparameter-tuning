# Analysis

## Part 1 — GridSearchCV

The hyperparameters that had the largest impact on the F1 score were max_depth and n_estimators. Increasing max_depth significantly improved performance, indicating that deeper trees were better at capturing patterns in the data. The number of estimators also contributed to performance, but its impact was less significant compared to max_depth.

The heatmap shows that performance improves as model complexity increases, especially with higher max_depth values. However, the performance begins to plateau after a certain point (around max_depth = 20 and n_estimators = 100), indicating a clear “sweet spot” where increasing complexity further does not provide meaningful gains.

The model appears to be slightly at risk of overfitting, as deeper trees perform better but may capture noise from the training data. This is supported by the drop in F1 score from cross-validation to the test set.

I aggregated across min_samples_split to better understand overall trends in model performance rather than focusing on a single configuration.

---

## Part 2 — Nested Cross-Validation

The Decision Tree model shows a larger gap between the inner best_score_ and the outer nested cross-validation score compared to the Random Forest. Specifically, the Decision Tree has a gap of approximately 0.0078, while the Random Forest has a much smaller gap of about 0.0018. This is expected because decision trees are high-variance models that are more sensitive to the specific training data, making them more prone to selection bias during hyperparameter tuning. In contrast, random forests reduce variance through bagging, resulting in more stable and reliable performance.

The GridSearchCV.best_score_ from Part 1 is relatively trustworthy for the Random Forest due to its very small gap, which indicates minimal optimistic bias. However, for the Decision Tree, the larger gap suggests that the inner cross-validation slightly overestimates performance, making it less reliable.

This connects directly to the Module 5 Week A concept of held-out test sets: data that is used to make decisions (such as hyperparameter tuning) cannot also be used to evaluate those decisions. Nested cross-validation solves this problem by introducing an outer loop that acts as an unbiased evaluation set, ensuring that model performance is measured on data that was not used during hyperparameter selection.