import csv
import os

import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.clean_reviews import clean_text


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_PATH = (
    "data/amazon_reviews/reviews.csv"
)

MODEL_PATH = (
    "models/review_sentiment.joblib"
)

METRICS_PATH = (
    "results/sentiment_metrics.csv"
)

CONFUSION_MATRIX_PATH = (
    "results/sentiment_confusion_matrix.csv"
)

TERMS_PATH = (
    "results/informative_terms.csv"
)

RANDOM_STATE = 42

TEST_SIZE = 0.20

TOP_TERMS = 20


# ============================================================
# 2. CHECK DATASET
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )


os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "results",
    exist_ok=True
)


# ============================================================
# 3. LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH
)

print("=" * 70)
print("DAY 12: CLASSICAL SENTIMENT ANALYSIS")
print("=" * 70)

print(
    "Dataset shape:",
    df.shape
)

print(
    "\nColumns:",
    df.columns.tolist()
)


# ============================================================
# 4. CHECK COLUMNS
# ============================================================

required_columns = {
    "review",
    "label"
}

missing_columns = (
    required_columns
    - set(df.columns)
)

if missing_columns:

    raise ValueError(
        "Dataset is missing columns: "
        f"{missing_columns}"
    )


# ============================================================
# 5. CLEAN DATA
# ============================================================

df = df[
    [
        "review",
        "label"
    ]
].copy()


df = df.dropna(
    subset=[
        "review",
        "label"
    ]
)


df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)


df = df.dropna(
    subset=["label"]
)


df["label"] = (
    df["label"]
    .astype(int)
)


# Keep binary labels only.
df = df[
    df["label"].isin(
        [0, 1]
    )
]


print(
    "\nLabel distribution:"
)

print(
    df["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(

        df["review"],

        df["label"],

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=df["label"]
    )
)


print(
    "\nTraining reviews:",
    len(X_train)
)

print(
    "Testing reviews :",
    len(X_test)
)


# ============================================================
# 7. CREATE COMPLETE NLP PIPELINE
# ============================================================

pipeline = Pipeline([
    (
        "tfidf",

        TfidfVectorizer(

            preprocessor=clean_text,

            lowercase=False,

            max_features=20000,

            ngram_range=(
                1,
                2
            ),

            min_df=2,

            max_df=0.95,

            sublinear_tf=True
        )
    ),

    (
        "classifier",

        LogisticRegression(

            max_iter=1000,

            random_state=RANDOM_STATE,

            class_weight="balanced"
        )
    )
])


# ============================================================
# 8. TRAIN
# ============================================================

print("\nTraining sentiment model...")


pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# 9. PREDICT
# ============================================================

predictions = pipeline.predict(
    X_test
)


# ============================================================
# 10. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


print("\n" + "=" * 70)
print("SENTIMENT RESULTS")
print("=" * 70)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1-score : {f1:.4f}"
)


print(
    "\nClassification report:"
)

print(
    classification_report(

        y_test,

        predictions,

        target_names=[
            "Negative",
            "Positive"
        ],

        zero_division=0
    )
)


# ============================================================
# 11. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame([
    {
        "model": (
            "TF-IDF + Logistic Regression"
        ),

        "accuracy": round(
            accuracy,
            4
        ),

        "precision": round(
            precision,
            4
        ),

        "recall": round(
            recall,
            4
        ),

        "f1": round(
            f1,
            4
        )
    }
])


metrics_df.to_csv(
    METRICS_PATH,
    index=False
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)


confusion_df = pd.DataFrame(

    cm,

    index=[
        "Actual Negative",
        "Actual Positive"
    ],

    columns=[
        "Predicted Negative",
        "Predicted Positive"
    ]
)


print(
    "\nConfusion matrix:"
)

print(confusion_df)


confusion_df.to_csv(
    CONFUSION_MATRIX_PATH
)


# ============================================================
# 13. EXTRACT INFORMATIVE TERMS
# ============================================================

vectorizer = pipeline.named_steps[
    "tfidf"
]

classifier = pipeline.named_steps[
    "classifier"
]


feature_names = (
    vectorizer
    .get_feature_names_out()
)


coefficients = (
    classifier.coef_[0]
)


# Smallest coefficients -> negative sentiment
negative_indices = (
    coefficients.argsort()[
        :TOP_TERMS
    ]
)


# Largest coefficients -> positive sentiment
positive_indices = (
    coefficients.argsort()[
        -TOP_TERMS:
    ][::-1]
)


informative_terms = []


for index in positive_indices:

    informative_terms.append({
        "sentiment": "positive",

        "term": (
            feature_names[index]
        ),

        "coefficient": round(
            float(
                coefficients[index]
            ),
            6
        )
    })


for index in negative_indices:

    informative_terms.append({
        "sentiment": "negative",

        "term": (
            feature_names[index]
        ),

        "coefficient": round(
            float(
                coefficients[index]
            ),
            6
        )
    })


terms_df = pd.DataFrame(
    informative_terms
)


terms_df.to_csv(
    TERMS_PATH,
    index=False
)


print(
    "\nTop positive terms:"
)

for index in positive_indices:

    print(
        f"{feature_names<30}"
        f"{coefficients.4f}"
    )


print(
    "\nTop negative terms:"
)

for index in negative_indices:

    print(
        f"{feature_names<30}"
        f"{coefficients.4f}"
    )


# ============================================================
# 14. SAVE PIPELINE
# ============================================================

joblib.dump(
    pipeline,
    MODEL_PATH
)


print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    "Sentiment model :",
    MODEL_PATH
)

print(
    "Metrics         :",
    METRICS_PATH
)

print(
    "Confusion matrix:",
    CONFUSION_MATRIX_PATH
)

print(
    "Informative terms:",
    TERMS_PATH
)