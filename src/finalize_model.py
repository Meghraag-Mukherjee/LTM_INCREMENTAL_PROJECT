import os
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from xgboost import XGBClassifier

from src.features import add_engineered_features


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_PATH = (
    "data/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

DAY3_MODEL_PATH = "models/churn_v3.joblib"

FINAL_MODEL_PATH = "models/churn_final.joblib"

COMPARISON_PATH = (
    "models/day4_model_comparison.csv"
)

RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("DAY 4: FINAL RETAILMIND CHURN MODEL")
print("=" * 70)

print("Original dataset shape:", df.shape)


# ============================================================
# 3. BASIC CLEANING
# ============================================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

if "customerID" in df.columns:
    df = df.drop(
        columns=["customerID"]
    )


# ============================================================
# 4. X AND y
# ============================================================

X = df.drop(
    columns=["Churn"]
).copy()

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

if y.isnull().any():
    raise ValueError(
        "Churn must contain only Yes and No."
    )


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )
)

print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))


# ============================================================
# 6. DETERMINE COLUMNS AFTER FEATURE ENGINEERING
# ============================================================

X_train_engineered = add_engineered_features(
    X_train
)

numerical_columns = (
    X_train_engineered
    .select_dtypes(include=["number"])
    .columns
    .tolist()
)

categorical_columns = (
    X_train_engineered
    .select_dtypes(
        include=["object", "category", "bool"]
    )
    .columns
    .tolist()
)

print("\nNumber of numerical columns:",
      len(numerical_columns))

print("Number of categorical columns:",
      len(categorical_columns))

print("\nEngineered columns added:")

original_columns = set(X_train.columns)

engineered_columns = [
    column
    for column in X_train_engineered.columns
    if column not in original_columns
]

for column in engineered_columns:
    print("-", column)


# ============================================================
# 7. FEATURE ENGINEERING TRANSFORMER
# ============================================================

feature_engineer = FunctionTransformer(
    add_engineered_features,
    validate=False
)


# ============================================================
# 8. PREPROCESSING
# ============================================================

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
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# ============================================================
# 9. PCA
# ============================================================

# Retain 95% of the transformed feature variance.
pca = PCA(
    n_components=0.95,
    random_state=RANDOM_STATE
)


# ============================================================
# 10. CLASS IMBALANCE WEIGHT
# ============================================================

negative_count = int(
    (y_train == 0).sum()
)

positive_count = int(
    (y_train == 1).sum()
)

scale_pos_weight = (
    negative_count /
    positive_count
)

print(
    "\nXGBoost scale_pos_weight:",
    round(scale_pos_weight, 4)
)


# ============================================================
# 11. RANDOM FOREST
# ============================================================

random_forest_pipeline = Pipeline(
    steps=[
        (
            "feature_engineering",
            feature_engineer
        ),
        (
            "preprocessor",
            preprocessor
        ),
        (
            "pca",
            pca
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# 12. XGBOOST
# ============================================================

xgboost_pipeline = Pipeline(
    steps=[
        (
            "feature_engineering",
            feature_engineer
        ),
        (
            "preprocessor",
            preprocessor
        ),
        (
            "pca",
            pca
        ),
        (
            "classifier",
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                min_child_weight=2,
                subsample=0.80,
                colsample_bytree=0.80,
                reg_alpha=0.10,
                reg_lambda=1.0,
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# 13. LOAD DAY 3 CHAMPION
# ============================================================

if not os.path.exists(DAY3_MODEL_PATH):
    raise FileNotFoundError(
        "models/churn_v3.joblib was not found. "
        "Run Day 3 tune_models.py first."
    )

day3_model = joblib.load(
    DAY3_MODEL_PATH
)


# ============================================================
# 14. ALL MODELS
# ============================================================

models = {
    "Day 3 Champion": day3_model,
    "Random Forest + Features + PCA":
        random_forest_pipeline,
    "XGBoost + Features + PCA":
        xgboost_pipeline
}


# ============================================================
# 15. CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


# ============================================================
# 16. TRAIN AND COMPARE
# ============================================================

comparison_results = []

trained_models = {}

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print("TRAINING:", model_name)
    print("=" * 70)

    cv_results = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True
    )

    cv_train_recall = (
        cv_results["train_recall"].mean()
    )

    cv_validation_recall = (
        cv_results["test_recall"].mean()
    )

    cv_roc_auc = (
        cv_results["test_roc_auc"].mean()
    )

    recall_gap = (
        cv_train_recall -
        cv_validation_recall
    )

    # Train on all training data after CV.
    model.fit(
        X_train,
        y_train
    )

    trained_models[model_name] = model

    y_pred = model.predict(
        X_test
    )

    # Probability needed for ROC-AUC.
    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    test_accuracy = accuracy_score(
        y_test,
        y_pred
    )

    test_precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    test_roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    comparison_results.append({
        "Model": model_name,
        "CV Train Recall": round(
            cv_train_recall,
            4
        ),
        "CV Validation Recall": round(
            cv_validation_recall,
            4
        ),
        "CV Recall Gap": round(
            recall_gap,
            4
        ),
        "CV ROC-AUC": round(
            cv_roc_auc,
            4
        ),
        "Test Accuracy": round(
            test_accuracy,
            4
        ),
        "Test Precision": round(
            test_precision,
            4
        ),
        "Test Recall": round(
            test_recall,
            4
        ),
        "Test F1": round(
            test_f1,
            4
        ),
        "Test ROC-AUC": round(
            test_roc_auc,
            4
        )
    })

    print(
        f"CV train recall      : "
        f"{cv_train_recall:.4f}"
    )

    print(
        f"CV validation recall : "
        f"{cv_validation_recall:.4f}"
    )

    print(
        f"CV recall gap        : "
        f"{recall_gap:.4f}"
    )

    print(
        f"CV ROC-AUC           : "
        f"{cv_roc_auc:.4f}"
    )

    print("\nHold-out test results:")

    print(
        f"Accuracy  : {test_accuracy:.4f}"
    )

    print(
        f"Precision : {test_precision:.4f}"
    )

    print(
        f"Recall    : {test_recall:.4f}"
    )

    print(
        f"F1-score  : {test_f1:.4f}"
    )

    print(
        f"ROC-AUC   : {test_roc_auc:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Churn",
                "Churn"
            ],
            zero_division=0
        )
    )


# ============================================================
# 17. CREATE COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(
    comparison_results
)

# The champion is selected using CV validation recall.
# CV ROC-AUC and CV recall gap act as tie-breakers.
results_df = results_df.sort_values(
    by=[
        "CV Validation Recall",
        "CV ROC-AUC",
        "CV Recall Gap"
    ],
    ascending=[
        False,
        False,
        True
    ]
).reset_index(drop=True)

print("\n" + "=" * 110)
print("FINAL MODEL COMPARISON")
print("=" * 110)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 18. SELECT FINAL CHAMPION
# ============================================================

final_model_name = (
    results_df.iloc[0]["Model"]
)

final_model = trained_models[
    final_model_name
]

print("\n" + "=" * 70)
print("FINAL PRODUCTION MODEL")
print("=" * 70)

print("Selected model:", final_model_name)

print(
    "CV validation recall:",
    results_df.iloc[0][
        "CV Validation Recall"
    ]
)

print(
    "Test recall:",
    results_df.iloc[0][
        "Test Recall"
    ]
)

print(
    "Test ROC-AUC:",
    results_df.iloc[0][
        "Test ROC-AUC"
    ]
)


# ============================================================
# 19. SAVE FINAL PIPELINE
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    final_model,
    FINAL_MODEL_PATH
)

results_df.to_csv(
    COMPARISON_PATH,
    index=False
)

print("\nFiles created:")

print(
    "Final model:",
    FINAL_MODEL_PATH
)

print(
    "Comparison:",
    COMPARISON_PATH
)