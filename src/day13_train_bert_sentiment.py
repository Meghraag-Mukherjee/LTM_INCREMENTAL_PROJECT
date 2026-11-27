import os

import joblib
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed
)


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_PATH = "data/amazon_reviews/reviews.csv"

DAY12_MODEL_PATH = (
    "models/review_sentiment.joblib"
)

MODEL_DIRECTORY = (
    "models/bert_sentiment"
)

RESULTS_DIRECTORY = "results"

METRICS_PATH = (
    "results/bert_sentiment_metrics.csv"
)

REPORT_PATH = (
    "results/bert_sentiment_report.csv"
)

CONFUSION_MATRIX_PATH = (
    "results/bert_sentiment_confusion_matrix.csv"
)

COMPARISON_PATH = (
    "results/bert_vs_classical.csv"
)

CHECKPOINT_DIRECTORY = (
    "results/bert_sentiment_checkpoints"
)

MODEL_NAME = "bert-base-uncased"

RANDOM_STATE = 42

MAX_LENGTH = 256

TEST_SIZE = 0.20

TRAINING_EPOCHS = 2

TRAIN_BATCH_SIZE = 8

EVALUATION_BATCH_SIZE = 16


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

set_seed(RANDOM_STATE)


# ============================================================
# 3. CHECK DATA
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )


os.makedirs(
    MODEL_DIRECTORY,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIRECTORY,
    exist_ok=True
)


# ============================================================
# 4. LOAD AND CLEAN REVIEWS
# ============================================================

reviews_df = pd.read_csv(
    DATA_PATH
)


required_columns = {
    "review",
    "label"
}


missing_columns = (
    required_columns
    - set(reviews_df.columns)
)


if missing_columns:
    raise ValueError(
        "Dataset is missing columns: "
        f"{missing_columns}"
    )


reviews_df = reviews_df[
    [
        "review",
        "label"
    ]
].copy()


reviews_df = reviews_df.dropna(
    subset=[
        "review",
        "label"
    ]
)


reviews_df["review"] = (
    reviews_df["review"]
    .astype(str)
    .str.strip()
)


reviews_df["label"] = pd.to_numeric(
    reviews_df["label"],
    errors="coerce"
)


reviews_df = reviews_df.dropna(
    subset=["label"]
)


reviews_df["label"] = (
    reviews_df["label"]
    .astype(int)
)


reviews_df = reviews_df[
    reviews_df["label"].isin(
        [0, 1]
    )
].copy()


reviews_df = reviews_df[
    reviews_df["review"].str.len() > 0
].reset_index(drop=True)


print("=" * 70)
print("DAY 13: BERT SENTIMENT CLASSIFICATION")
print("=" * 70)

print(
    "Dataset size:",
    len(reviews_df)
)

print(
    "\nLabel distribution:"
)

print(
    reviews_df["label"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

train_df, test_df = train_test_split(
    reviews_df,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=reviews_df["label"]
)


train_df = train_df.reset_index(
    drop=True
)

test_df = test_df.reset_index(
    drop=True
)


print(
    "\nTraining reviews:",
    len(train_df)
)

print(
    "Testing reviews :",
    len(test_df)
)


# ============================================================
# 6. LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 7. PYTORCH DATASET
# ============================================================

class ReviewDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length
    ):

        self.texts = list(texts)

        self.labels = list(labels)

        self.tokenizer = tokenizer

        self.max_length = max_length


    def __len__(self):

        return len(self.texts)


    def __getitem__(self, index):

        encoded_review = self.tokenizer(
            self.texts[index],

            truncation=True,

            max_length=self.max_length
        )


        encoded_review["labels"] = int(
            self.labels[index]
        )


        return encoded_review


train_dataset = ReviewDataset(
    texts=train_df["review"],
    labels=train_df["label"],
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)


test_dataset = ReviewDataset(
    texts=test_df["review"],
    labels=test_df["label"],
    tokenizer=tokenizer,
    max_length=MAX_LENGTH
)


# ============================================================
# 8. DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# 9. LOAD PRETRAINED BERT
# ============================================================

id_to_label = {
    0: "NEGATIVE",
    1: "POSITIVE"
}


label_to_id = {
    "NEGATIVE": 0,
    "POSITIVE": 1
}


model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_NAME,

        num_labels=2,

        id2label=id_to_label,

        label2id=label_to_id
    )
)


# ============================================================
# 10. METRIC FUNCTION
# ============================================================

def calculate_metrics(evaluation_prediction):

    logits, labels = evaluation_prediction

    predictions = np.argmax(
        logits,
        axis=-1
    )


    return {
        "accuracy": accuracy_score(
            labels,
            predictions
        ),

        "precision": precision_score(
            labels,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            labels,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            labels,
            predictions,
            zero_division=0
        )
    }


# ============================================================
# 11. TRAINING SETTINGS
# ============================================================

training_arguments = TrainingArguments(
    output_dir=CHECKPOINT_DIRECTORY,

    learning_rate=2e-5,

    per_device_train_batch_size=(
        TRAIN_BATCH_SIZE
    ),

    per_device_eval_batch_size=(
        EVALUATION_BATCH_SIZE
    ),

    num_train_epochs=TRAINING_EPOCHS,

    weight_decay=0.01,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="steps",

    logging_steps=50,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    save_total_limit=1,

    report_to="none",

    seed=RANDOM_STATE,

    fp16=torch.cuda.is_available()
)


# ============================================================
# 12. TRAINER
# ============================================================

trainer = Trainer(
    model=model,

    args=training_arguments,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    data_collator=data_collator,

    processing_class=tokenizer,

    compute_metrics=calculate_metrics
)


# ============================================================
# 13. FINE-TUNE BERT
# ============================================================

print(
    "\nDevice:",
    "GPU"
    if torch.cuda.is_available()
    else "CPU"
)


print(
    "\nFine-tuning BERT..."
)


trainer.train()


# ============================================================
# 14. TEST PREDICTIONS
# ============================================================

prediction_output = trainer.predict(
    test_dataset
)


test_predictions = np.argmax(
    prediction_output.predictions,
    axis=-1
)


test_labels = (
    test_df["label"]
    .to_numpy()
)


bert_accuracy = accuracy_score(
    test_labels,
    test_predictions
)


bert_precision = precision_score(
    test_labels,
    test_predictions,
    zero_division=0
)


bert_recall = recall_score(
    test_labels,
    test_predictions,
    zero_division=0
)


bert_f1 = f1_score(
    test_labels,
    test_predictions,
    zero_division=0
)


print("\n" + "=" * 70)
print("BERT SENTIMENT RESULTS")
print("=" * 70)

print(
    f"Accuracy : {bert_accuracy:.4f}"
)

print(
    f"Precision: {bert_precision:.4f}"
)

print(
    f"Recall   : {bert_recall:.4f}"
)

print(
    f"F1-score : {bert_f1:.4f}"
)


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    test_labels,
    test_predictions,

    target_names=[
        "Negative",
        "Positive"
    ],

    output_dict=True,

    zero_division=0
)


report_df = (
    pd.DataFrame(report)
    .transpose()
)


report_df.to_csv(
    REPORT_PATH
)


print(
    "\nClassification report:"
)

print(
    classification_report(
        test_labels,
        test_predictions,

        target_names=[
            "Negative",
            "Positive"
        ],

        zero_division=0
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

sentiment_confusion_matrix = (
    confusion_matrix(
        test_labels,
        test_predictions
    )
)


confusion_df = pd.DataFrame(
    sentiment_confusion_matrix,

    index=[
        "Actual Negative",
        "Actual Positive"
    ],

    columns=[
        "Predicted Negative",
        "Predicted Positive"
    ]
)


confusion_df.to_csv(
    CONFUSION_MATRIX_PATH
)


print(
    "\nConfusion matrix:"
)

print(confusion_df)


# ============================================================
# 17. SAVE BERT METRICS
# ============================================================

bert_metrics_df = pd.DataFrame([
    {
        "model": "BERT",

        "checkpoint": MODEL_NAME,

        "accuracy": round(
            bert_accuracy,
            4
        ),

        "precision": round(
            bert_precision,
            4
        ),

        "recall": round(
            bert_recall,
            4
        ),

        "f1": round(
            bert_f1,
            4
        )
    }
])


bert_metrics_df.to_csv(
    METRICS_PATH,
    index=False
)


# ============================================================
# 18. COMPARE WITH DAY 12 MODEL
# ============================================================

comparison_rows = [
    {
        "model": "BERT",

        "accuracy": round(
            bert_accuracy,
            4
        ),

        "precision": round(
            bert_precision,
            4
        ),

        "recall": round(
            bert_recall,
            4
        ),

        "f1": round(
            bert_f1,
            4
        )
    }
]


if os.path.exists(
    DAY12_MODEL_PATH
):

    day12_model = joblib.load(
        DAY12_MODEL_PATH
    )


    day12_predictions = (
        day12_model.predict(
            test_df["review"]
        )
    )


    comparison_rows.append({
        "model": (
            "Day-12 TF-IDF + "
            "Logistic Regression"
        ),

        "accuracy": round(
            accuracy_score(
                test_labels,
                day12_predictions
            ),
            4
        ),

        "precision": round(
            precision_score(
                test_labels,
                day12_predictions,
                zero_division=0
            ),
            4
        ),

        "recall": round(
            recall_score(
                test_labels,
                day12_predictions,
                zero_division=0
            ),
            4
        ),

        "f1": round(
            f1_score(
                test_labels,
                day12_predictions,
                zero_division=0
            ),
            4
        )
    })


comparison_df = pd.DataFrame(
    comparison_rows
)


comparison_df.to_csv(
    COMPARISON_PATH,
    index=False
)


print(
    "\nBERT versus Day-12 baseline:"
)

print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# 19. SAVE MODEL AND TOKENIZER
# ============================================================

trainer.save_model(
    MODEL_DIRECTORY
)


tokenizer.save_pretrained(
    MODEL_DIRECTORY
)


print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    "BERT sentiment model:",
    MODEL_DIRECTORY
)

print(
    "Metrics:",
    METRICS_PATH
)

print(
    "Classification report:",
    REPORT_PATH
)

print(
    "Confusion matrix:",
    CONFUSION_MATRIX_PATH
)

print(
    "Baseline comparison:",
    COMPARISON_PATH
)