# train.py
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib

DATA_PATH = Path("loan.csv")
MODEL_PATH = Path("loan_approval_model.pkl")

# 1) Load
df = pd.read_csv(DATA_PATH)

# 2) Basic cleaning
# Strip spaces in column names
df.columns = [c.strip() for c in df.columns]

# Drop ID if present
id_cols = [c for c in df.columns if c.lower() in ["loan_id", "id"]]
df = df.drop(columns=id_cols, errors="ignore")

# Map target to 0/1 if it's Y/N or similar
target_col_candidates = [c for c in df.columns if c.lower() in ["loan_status","approved","loan_approved"]]
assert len(target_col_candidates) == 1, f"Couldn't identify target column. Found: {target_col_candidates}"
target_col = target_col_candidates[0]

# Handle common target encodings
y_raw = df[target_col].astype(str).str.upper().str.strip()
y = y_raw.replace({"Y":1, "N":0, "YES":1, "NO":0, "APPROVED":1, "REJECTED":0})
if not set(y.unique()).issubset({0,1}):
    # if already numeric, try converting
    y = pd.to_numeric(df[target_col], errors="coerce")
    assert set(y.dropna().unique()).issubset({0,1}), "Target must be binary 0/1 or Y/N"
df = df.drop(columns=[target_col])

# 3) Split columns by dtype (you can also hardcode lists below)
categorical_cols = []
numeric_cols = []
for c in df.columns:
    if df[c].dtype == "object":
        categorical_cols.append(c)
    else:
        numeric_cols.append(c)

# Some known numeric columns may be read as object (e.g., 'Dependents' like '3+')
# Convert reasonable numeric-like columns
for col in ["Dependents", "Loan_Amount_Term", "Credit_History"]:
    if col in df.columns and df[col].dtype == "object":
        df[col] = df[col].replace({"3+": "3", "nan": np.nan}).astype("float64")
        if col not in numeric_cols:
            numeric_cols.append(col)
            if col in categorical_cols:
                categorical_cols.remove(col)

# 4) Preprocess pipelines
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ],
    remainder="drop"
)

# 5) Model – start with LogisticRegression (fast, interpretable)
clf = LogisticRegression(max_iter=500, n_jobs=None)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", clf)
])

# 6) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y
)

# 7) Fit
pipe.fit(X_train, y_train)

# 8) Evaluate
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9) Save model (pipeline includes preprocessing)
joblib.dump({
    "pipeline": pipe,
    "feature_columns": df.columns.tolist(),
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "target_name": target_col
}, MODEL_PATH)

print(f"\nSaved model to: {MODEL_PATH.resolve()}")
