import os
import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    learning_curve
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

MODEL_DIRECTORY = "models"

MODEL_PATH = "models/churn_v3.joblib"

RESULTS_PATH = "models/day3_tuning_results.csv"

LEARNING_CURVE_PATH = "models/day3_learning_curve.csv"

RANDOM_STATE = 42


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nTarget distribution:")
print(df["Churn"].value_counts())

print("\nMissing values before cleaning:")
print(df.isnull().sum())


# ============================================================
# 3. CLEAN DATA
# ============================================================

# TotalCharges may contain blank strings.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# customerID is an identifier, not a predictive feature.
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])


# ============================================================
# 4. CREATE X AND y
# ============================================================

X = df.drop(columns=["Churn"])

y = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

if y.isnull().any():
    raise ValueError(
        "The Churn column contains values other than 'Yes' and 'No'."
    )


# ============================================================
# 5. IDENTIFY COLUMN TYPES
# ============================================================

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)


# ============================================================
# 6. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape :", X_test.shape)


# ============================================================
# 7. PREPROCESSING PIPELINES
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


# ============================================================
# 8. CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# 9. CANDIDATE MODELS AND PARAMETER GRIDS
# ============================================================

model_configs = {
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),

        "parameters": {
            "classifier__criterion": [
                "gini",
                "entropy"
            ],

            "classifier__max_depth": [
                3,
                5,
                8,
                None
            ],

            "classifier__min_samples_split": [
                2,
                5,
                10
            ],

            "classifier__min_samples_leaf": [
                1,
                2,
                4
            ],

            "classifier__class_weight": [
                None,
                "balanced"
            ]
        }
    },

    "Random Forest": {
        "estimator": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "parameters": {
            "classifier__n_estimators": [
                100,
                200
            ],

            "classifier__max_depth": [
                5,
                10,
                None
            ],

            "classifier__min_samples_split": [
                2,
                5
            ],

            "classifier__min_samples_leaf": [
                1,
                2
            ],

            "classifier__class_weight": [
                None,
                "balanced"
            ]
        }
    },

    "SVM": {
        "estimator": SVC(
            probability=True,
            random_state=RANDOM_STATE
        ),

        "parameters": {
            "classifier__C": [
                0.1,
                1,
                10
            ],

            "classifier__kernel": [
                "linear",
                "rbf"
            ],

            "classifier__gamma": [
                "scale",
                "auto"
            ],

            "classifier__class_weight": [
                None,
                "balanced"
            ]
        }
    }
}


# ============================================================
# 10. GRID SEARCH FOR EACH MODEL
# ============================================================

comparison_results = []

tuned_models = {}

for model_name, configuration in model_configs.items():

    print("\n" + "=" * 70)
    print(f"TUNING: {model_name}")
    print("=" * 70)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                configuration["estimator"]
            )
        ]
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=configuration["parameters"],
        scoring="recall",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
        refit=True
    )

    grid_search.fit(
        X_train,
        y_train
    )

    best_index = grid_search.best_index_

    mean_train_recall = grid_search.cv_results_[
        "mean_train_score"
    ][best_index]

    mean_validation_recall = grid_search.cv_results_[
        "mean_test_score"
    ][best_index]

    score_gap = (
        mean_train_recall -
        mean_validation_recall
    )

    if (
        mean_train_recall < 0.70
        and mean_validation_recall < 0.70
    ):
        diagnosis = "Possible high bias / underfitting"

    elif score_gap > 0.10:
        diagnosis = "Possible high variance / overfitting"

    else:
        diagnosis = "Reasonable bias-variance balance"

    best_pipeline = grid_search.best_estimator_

    tuned_models[model_name] = best_pipeline

    y_pred = best_pipeline.predict(X_test)

    y_probability = best_pipeline.predict_proba(
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
        "Best Parameters": str(
            grid_search.best_params_
        ),
        "CV Train Recall": round(
            mean_train_recall,
            4
        ),
        "CV Validation Recall": round(
            mean_validation_recall,
            4
        ),
        "Train Validation Gap": round(
            score_gap,
            4
        ),
        "Diagnosis": diagnosis,
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

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(
        "\nBest CV validation recall:",
        round(grid_search.best_score_, 4)
    )

    print(
        "Mean CV training recall:",
        round(mean_train_recall, 4)
    )

    print(
        "Mean CV validation recall:",
        round(mean_validation_recall, 4)
    )

    print(
        "Train-validation gap:",
        round(score_gap, 4)
    )

    print("Diagnosis:", diagnosis)

    print("\nHold-out test results:")
    print(f"Accuracy : {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall   : {test_recall:.4f}")
    print(f"F1-score : {test_f1:.4f}")
    print(f"ROC-AUC  : {test_roc_auc:.4f}")

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
# 11. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    comparison_results
)

# Select primarily by validation recall.
# ROC-AUC is used as the tie-breaker.
results_df = results_df.sort_values(
    by=[
        "CV Validation Recall",
        "Test ROC-AUC"
    ],
    ascending=[
        False,
        False
    ]
).reset_index(drop=True)

print("\n" + "=" * 100)
print("TUNED MODEL COMPARISON")
print("=" * 100)

display_columns = [
    "Model",
    "CV Train Recall",
    "CV Validation Recall",
    "Train Validation Gap",
    "Diagnosis",
    "Test Accuracy",
    "Test Precision",
    "Test Recall",
    "Test F1",
    "Test ROC-AUC"
]

print(
    results_df[
        display_columns
    ].to_string(index=False)
)


# ============================================================
# 12. SELECT CHAMPION
# ============================================================

champion_name = results_df.iloc[0]["Model"]

champion_model = tuned_models[
    champion_name
]

print("\n" + "=" * 70)
print("CHAMPION MODEL")
print("=" * 70)

print("Champion:", champion_name)

print(
    "CV validation recall:",
    results_df.iloc[0]["CV Validation Recall"]
)

print(
    "Test recall:",
    results_df.iloc[0]["Test Recall"]
)

print(
    "Test ROC-AUC:",
    results_df.iloc[0]["Test ROC-AUC"]
)

print(
    "Best parameters:",
    results_df.iloc[0]["Best Parameters"]
)


# ============================================================
# 13. LEARNING CURVE FOR CHAMPION
# ============================================================

print("\nGenerating champion learning curve data...")

train_sizes, train_scores, validation_scores = learning_curve(
    estimator=champion_model,
    X=X_train,
    y=y_train,
    cv=cv,
    scoring="recall",
    train_sizes=[
        0.20,
        0.40,
        0.60,
        0.80,
        1.00
    ],
    n_jobs=-1,
    shuffle=True,
    random_state=RANDOM_STATE
)

mean_train_scores = train_scores.mean(axis=1)

mean_validation_scores = validation_scores.mean(
    axis=1
)

learning_curve_df = pd.DataFrame({
    "Training Examples": train_sizes,
    "Mean Training Recall": mean_train_scores.round(4),
    "Mean Validation Recall": mean_validation_scores.round(4),
    "Recall Gap": (
        mean_train_scores -
        mean_validation_scores
    ).round(4)
})

print("\nChampion learning curve:")
print(
    learning_curve_df.to_string(
        index=False
    )
)


# ============================================================
# 14. SAVE MODEL AND RESULTS
# ============================================================

os.makedirs(
    MODEL_DIRECTORY,
    exist_ok=True
)

joblib.dump(
    champion_model,
    MODEL_PATH
)

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

learning_curve_df.to_csv(
    LEARNING_CURVE_PATH,
    index=False
)

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print("Tuned champion model:", MODEL_PATH)
print("Tuning comparison    :", RESULTS_PATH)
print("Learning curve data  :", LEARNING_CURVE_PATH)