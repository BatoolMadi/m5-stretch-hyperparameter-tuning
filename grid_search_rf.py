# 1. Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Load data
df = pd.read_csv("data/telecom_churn.csv")

# 3. Prepare features and target
X = df.drop(["churned", "customer_id"], axis=1)
y = df["churned"]

# Handle categorical variables
X = pd.get_dummies(X, drop_first=True)

# 4. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 5. Model
rf = RandomForestClassifier(random_state=42)

# 6. Parameter Grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

# 7. Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 8. GridSearch
grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='f1',
    cv=cv,
    n_jobs=-1
)

# 9. Train
grid.fit(X_train, y_train)

# 10. Best Results
print("Best Params:", grid.best_params_)
print("Best F1 Score (CV):", grid.best_score_)

# 11. Evaluate on Test Set (extra but important)
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("Test F1 Score:", f1_score(y_test, y_pred))

# 12. Heatmap
results = pd.DataFrame(grid.cv_results_)

# We aggregate across min_samples_split to visualize overall trends
pivot_table = results.pivot_table(
    values='mean_test_score',
    index='param_max_depth',
    columns='param_n_estimators',
    aggfunc='mean'
)

# Sort for better visualization
pivot_table = pivot_table.sort_index()

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="viridis")

plt.title("F1 Score Heatmap (max_depth vs n_estimators)")
plt.xlabel("n_estimators")
plt.ylabel("max_depth")

plt.savefig("heatmap.png")
plt.show()