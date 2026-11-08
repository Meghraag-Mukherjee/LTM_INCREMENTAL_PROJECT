import os
import random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_PATH = (
    "data/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

CLASSICAL_MODEL_PATH = "models/churn_final.joblib"

ANN_MODEL_PATH = "models/churn_ann.keras"

PREPROCESSOR_PATH = "models/ann_preprocessor.joblib"

COMPARISON_PATH = "models/ann_comparison.csv"

TRAINING_HISTORY_PATH = "models/ann_training_history.csv"

RANDOM_STATE = 42


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)

np.random.seed(RANDOM_STATE)

tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 3. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("RETAILMIND ANN CHURN MODEL")
print("=" * 70)

print("Dataset shape:", df.shape)

print("\nTarget distribution:")

print(
    df["Churn"].value_counts()
)

print("\nTarget percentages:")

print(
    df["Churn"]
    .value_counts(normalize=True)
    .round(4)
)


# ============================================================
# 4. BASIC DATA CLEANING
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
# 5. SEPARATE FEATURES AND TARGET
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
        "The Churn column must contain only Yes and No."
    )


# ============================================================
# 6. TRAIN, VALIDATION AND TEST SPLIT
# ============================================================

# 80% development data and 20% final test data
X_development, X_test, y_development, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )
)

# Split the development set into training and validation sets
# 0.25 of 80% gives 20% of the complete dataset.
X_train, X_validation, y_train, y_validation = (
    train_test_split(
        X_development,
        y_development,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_development
    )
)

print("\nData split:")

print("Training rows  :", len(X_train))
print("Validation rows:", len(X_validation))
print("Testing rows   :", len(X_test))


# ============================================================
# 7. IDENTIFY COLUMN TYPES
# ============================================================

numerical_columns = (
    X_train
    .select_dtypes(include=["number"])
    .columns
    .tolist()
)

categorical_columns = (
    X_train
    .select_dtypes(
        include=["object", "category", "bool"]
    )
    .columns
    .tolist()
)

print("\nNumerical columns:")

print(numerical_columns)

print("\nCategorical columns:")

print(categorical_columns)


# ============================================================
# 8. CREATE ANN PREPROCESSOR
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
# 9. FIT PREPROCESSOR ONLY ON TRAINING DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_validation_processed = preprocessor.transform(
    X_validation
)

X_test_processed = preprocessor.transform(
    X_test
)

# TensorFlow works well with float32 arrays.
X_train_processed = X_train_processed.astype(
    "float32"
)

X_validation_processed = (
    X_validation_processed.astype("float32")
)

X_test_processed = X_test_processed.astype(
    "float32"
)

print(
    "\nANN input feature count:",
    X_train_processed.shape[1]
)


# ============================================================
# 10. HANDLE CLASS IMBALANCE
# ============================================================

classes = np.unique(y_train)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(class_label): float(weight)
    for class_label, weight
    in zip(classes, class_weights_array)
}

print("\nClass weights:")

print(class_weights)


# ============================================================
# 11. BUILD ANN
# ============================================================

input_feature_count = X_train_processed.shape[1]

ann_model = keras.Sequential([
    layers.Input(
        shape=(input_feature_count,),
        name="customer_features"
    ),

    layers.Dense(
        64,
        activation="relu",
        name="hidden_layer_1"
    ),

    layers.BatchNormalization(
        name="batch_normalization_1"
    ),

    layers.Dropout(
        0.30,
        name="dropout_1"
    ),

    layers.Dense(
        32,
        activation="relu",
        name="hidden_layer_2"
    ),

    layers.Dropout(
        0.20,
        name="dropout_2"
    ),

    layers.Dense(
        16,
        activation="relu",
        name="hidden_layer_3"
    ),

    layers.Dense(
        1,
        activation="sigmoid",
        name="churn_probability"
    )
])


# ============================================================
# 12. COMPILE ANN
# ============================================================

ann_model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="binary_crossentropy",

    metrics=[
        keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        keras.metrics.Precision(
            name="precision"
        ),

        keras.metrics.Recall(
            name="recall"
        ),

        keras.metrics.AUC(
            name="auc"
        )
    ]
)

print("\nANN architecture:")

ann_model.summary()


# ============================================================
# 13. CALLBACKS
# ============================================================

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    mode="min",
    restore_best_weights=True,
    verbose=1
)

reduce_learning_rate = (
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
)


# ============================================================
# 14. TRAIN ANN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING ANN")
print("=" * 70)

history = ann_model.fit(
    X_train_processed,
    y_train.to_numpy(),

    validation_data=(
        X_validation_processed,
        y_validation.to_numpy()
    ),

    epochs=100,
    batch_size=32,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        reduce_learning_rate
    ],

    verbose=1
)


# ============================================================
# 15. ANN PROBABILITIES AND PREDICTIONS
# ============================================================

ann_probabilities = (
    ann_model.predict(
        X_test_processed,
        verbose=0
    )
    .reshape(-1)
)

# Default binary classification threshold
ANN_THRESHOLD = 0.50

ann_predictions = (
    ann_probabilities >= ANN_THRESHOLD
).astype(int)


# ============================================================
# 16. ANN METRICS
# ============================================================

ann_accuracy = accuracy_score(
    y_test,
    ann_predictions
)

ann_precision = precision_score(
    y_test,
    ann_predictions,
    zero_division=0
)

ann_recall = recall_score(
    y_test,
    ann_predictions,
    zero_division=0
)

ann_f1 = f1_score(
    y_test,
    ann_predictions,
    zero_division=0
)

ann_roc_auc = roc_auc_score(
    y_test,
    ann_probabilities
)

print("\n" + "=" * 70)
print("ANN TEST RESULTS")
print("=" * 70)

print(f"Threshold : {ANN_THRESHOLD:.2f}")
print(f"Accuracy  : {ann_accuracy:.4f}")
print(f"Precision : {ann_precision:.4f}")
print(f"Recall    : {ann_recall:.4f}")
print(f"F1-score  : {ann_f1:.4f}")
print(f"ROC-AUC   : {ann_roc_auc:.4f}")

print("\nANN confusion matrix:")

print(
    confusion_matrix(
        y_test,
        ann_predictions
    )
)

print("\nANN classification report:")

print(
    classification_report(
        y_test,
        ann_predictions,
        target_names=[
            "No Churn",
            "Churn"
        ],
        zero_division=0
    )
)


# ============================================================
# 17. LOAD PHASE-1 CLASSICAL MODEL
# ============================================================

comparison_results = []

comparison_results.append({
    "Model": "Keras ANN",
    "Accuracy": round(ann_accuracy, 4),
    "Precision": round(ann_precision, 4),
    "Recall": round(ann_recall, 4),
    "F1": round(ann_f1, 4),
    "ROC-AUC": round(ann_roc_auc, 4)
})

if os.path.exists(CLASSICAL_MODEL_PATH):

    print("\n" + "=" * 70)
    print("EVALUATING PHASE-1 CLASSICAL CHAMPION")
    print("=" * 70)

    # Import this before loading if the Phase-1 pipeline
    # contains add_engineered_features.
    try:
        from src.features import add_engineered_features
    except ImportError:
        add_engineered_features = None

    classical_model = joblib.load(
        CLASSICAL_MODEL_PATH
    )

    classical_predictions = classical_model.predict(
        X_test
    )

    if hasattr(
        classical_model,
        "predict_proba"
    ):
        classical_probabilities = (
            classical_model.predict_proba(
                X_test
            )[:, 1]
        )

    elif hasattr(
        classical_model,
        "decision_function"
    ):
        decision_values = (
            classical_model.decision_function(
                X_test
            )
        )

        classical_probabilities = (
            1 /
            (
                1 +
                np.exp(-decision_values)
            )
        )

    else:
        classical_probabilities = (
            classical_predictions.astype(float)
        )

    classical_accuracy = accuracy_score(
        y_test,
        classical_predictions
    )

    classical_precision = precision_score(
        y_test,
        classical_predictions,
        zero_division=0
    )

    classical_recall = recall_score(
        y_test,
        classical_predictions,
        zero_division=0
    )

    classical_f1 = f1_score(
        y_test,
        classical_predictions,
        zero_division=0
    )

    classical_roc_auc = roc_auc_score(
        y_test,
        classical_probabilities
    )

    comparison_results.append({
        "Model": "Phase-1 Classical Champion",
        "Accuracy": round(
            classical_accuracy,
            4
        ),
        "Precision": round(
            classical_precision,
            4
        ),
        "Recall": round(
            classical_recall,
            4
        ),
        "F1": round(
            classical_f1,
            4
        ),
        "ROC-AUC": round(
            classical_roc_auc,
            4
        )
    })

    print(
        f"Accuracy  : {classical_accuracy:.4f}"
    )

    print(
        f"Precision : {classical_precision:.4f}"
    )

    print(
        f"Recall    : {classical_recall:.4f}"
    )

    print(
        f"F1-score  : {classical_f1:.4f}"
    )

    print(
        f"ROC-AUC   : {classical_roc_auc:.4f}"
    )

else:
    print(
        "\nWarning: models/churn_final.joblib "
        "was not found."
    )

    print(
        "The ANN was trained, but the Phase-1 "
        "comparison was skipped."
    )


# ============================================================
# 18. COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame(
    comparison_results
)

comparison_df = comparison_df.sort_values(
    by=[
        "Recall",
        "ROC-AUC"
    ],
    ascending=[
        False,
        False
    ]
).reset_index(drop=True)

print("\n" + "=" * 85)
print("ANN VS CLASSICAL ML")
print("=" * 85)

print(
    comparison_df.to_string(
        index=False
    )
)

winner = comparison_df.iloc[0]["Model"]

print("\nBest model by recall:", winner)


# ============================================================
# 19. SAVE EVERYTHING
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

ann_model.save(
    ANN_MODEL_PATH
)

joblib.dump(
    preprocessor,
    PREPROCESSOR_PATH
)

comparison_df.to_csv(
    COMPARISON_PATH,
    index=False
)

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    TRAINING_HISTORY_PATH,
    index=False
)

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print("ANN model       :", ANN_MODEL_PATH)
print("Preprocessor    :", PREPROCESSOR_PATH)
print("Comparison      :", COMPARISON_PATH)
print("Training history:", TRAINING_HISTORY_PATH)