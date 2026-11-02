import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)


# ==================================================
# 1. LOAD DATA
# ==================================================

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ==================================================
# 2. CLEAN DATA
# ==================================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# customerID is not useful for prediction
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])


# ==================================================
# 3. X AND y
# ==================================================

X = df.drop(columns=["Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# ==================================================
# 4. IDENTIFY COLUMN TYPES
# ==================================================

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()


# ==================================================
# 5. SAME HOLD-OUT SPLIT FOR ALL MODELS
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==================================================
# 6. DAY-1 PREPROCESSING
# ==================================================

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


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)


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


# ==================================================
# 7. CREATE 3 MODELS
# ==================================================

models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "SVM": SVC(
        probability=True,
        random_state=42
    )
}


# ==================================================
# 8. TRAIN AND EVALUATE
# ==================================================

results = []

trained_models = {}


for model_name, classifier in models.items():

    print(f"\nTraining {model_name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Probability for ROC-AUC
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "ROC-AUC": roc_auc
    })

    trained_models[model_name] = pipeline


# ==================================================
# 9. COMPARISON TABLE
# ==================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Recall",
    ascending=False
).reset_index(drop=True)


print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ==================================================
# 10. SELECT CHAMPION
# ==================================================

# Assignment says recall matters most for churn.
# Highest recall becomes the champion.

best_model_name = results_df.iloc[0]["Model"]

champion_model = trained_models[best_model_name]


print("\nChampion Model:", best_model_name)

print(
    "Champion Recall:",
    round(results_df.iloc[0]["Recall"], 4)
)


# ==================================================
# 11. SAVE CHAMPION
# ==================================================

os.makedirs("models", exist_ok=True)

CHAMPION_PATH = "models/champion_model.joblib"

joblib.dump(
    champion_model,
    CHAMPION_PATH
)

print(
    "\nChampion model saved to:",
    CHAMPION_PATH
)


# ==================================================
# 12. SAVE COMPARISON RESULTS
# ==================================================

results_df.to_csv(
    "models/day2_model_comparison.csv",
    index=False
)

print(
    "Comparison saved to:",
    "models/day2_model_comparison.csv"
)