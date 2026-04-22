# Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("data/telecom_churn.csv")

X = df.drop(["churned", "customer_id"], axis=1)
y = df["churned"]

X = pd.get_dummies(X, drop_first=True)

# Outer CV
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

# Inner CV
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -------------------------
# Random Forest
# -------------------------
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

rf_inner_scores = []
rf_outer_scores = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    rf = RandomForestClassifier(class_weight='balanced', random_state=42)

    grid = GridSearchCV(
        rf,
        rf_param_grid,
        scoring='f1',
        cv=inner_cv,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    # inner score
    rf_inner_scores.append(grid.best_score_)

    # outer score
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    rf_outer_scores.append(f1_score(y_test, y_pred))


# -------------------------
# Decision Tree
# -------------------------
dt_param_grid = {
    'max_depth': [3, 5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

dt_inner_scores = []
dt_outer_scores = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    dt = DecisionTreeClassifier(class_weight='balanced', random_state=42)

    grid = GridSearchCV(
        dt,
        dt_param_grid,
        scoring='f1',
        cv=inner_cv,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    # inner
    dt_inner_scores.append(grid.best_score_)

    # outer
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    dt_outer_scores.append(f1_score(y_test, y_pred))


# -------------------------
# Results
# -------------------------
results = pd.DataFrame({
    "Model": ["Random Forest", "Decision Tree"],
    "Inner Score (mean)": [
        np.mean(rf_inner_scores),
        np.mean(dt_inner_scores)
    ],
    "Outer Score (mean)": [
        np.mean(rf_outer_scores),
        np.mean(dt_outer_scores)
    ]
})

results["Gap (Inner - Outer)"] = results["Inner Score (mean)"] - results["Outer Score (mean)"]

print("\nNested CV Results:\n")
print(results)