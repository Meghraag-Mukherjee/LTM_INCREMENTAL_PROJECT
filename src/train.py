import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget balance:")
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True))


# --------------------------------------------------
# 2. Basic cleaning
# --------------------------------------------------

# TotalCharges can contain blank strings
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# customerID is only an identifier, so don't train on it
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])


# --------------------------------------------------
# 3. Separate X and y
# --------------------------------------------------

X = df.drop(columns=["Churn"])

# Convert Yes/No target into 1/0
y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# --------------------------------------------------
# 4. Detect numerical and categorical columns
# --------------------------------------------------

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)


# --------------------------------------------------
# 5. Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# --------------------------------------------------
# 6. Numerical preprocessing
# --------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# --------------------------------------------------
# 7. Categorical preprocessing
# --------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


# --------------------------------------------------
# 8. ColumnTransformer
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# --------------------------------------------------
# 9. Complete ML pipeline
# --------------------------------------------------

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# --------------------------------------------------
# 10. Train model
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# 11. Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

# Probability of class 1 = churn
y_probability = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 12. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n========== BASELINE RESULTS ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# --------------------------------------------------
# 13. Save trained pipeline
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/churn_pipeline.joblib"

joblib.dump(
    model,
    MODEL_PATH
)

print(f"\nModel saved to: {MODEL_PATH}")