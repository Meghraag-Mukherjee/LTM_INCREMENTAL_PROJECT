import json
import os

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from sklearn.model_selection import (
    train_test_split
)

from sklearn.preprocessing import (
    LabelEncoder
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
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

DATA_PATH = (
    "data/support_tickets/tickets.csv"
)

MODEL_DIRECTORY = (
    "models/bert_ticket_router"
)

METRICS_PATH = (
    "results/bert_ticket_metrics.csv"
)

REPORT_PATH = (
    "results/bert_ticket_report.csv"
)

CONFUSION_MATRIX_PATH = (
    "results/bert_ticket_confusion_matrix.csv"
)

LABELS_PATH = os.path.join(
    MODEL_DIRECTORY,
    "ticket_labels.json"
)

CHECKPOINT_DIRECTORY = (
    "results/bert_ticket_checkpoints"
)

MODEL_NAME = "bert-base-uncased"

RANDOM_STATE = 42

MAX_LENGTH = 256

TEST_SIZE = 0.20

TRAINING_EPOCHS = 3

TRAIN_BATCH_SIZE = 8

EVALUATION_BATCH_SIZE = 16


set_seed(RANDOM_STATE)


# ============================================================
# 2. LOAD TICKETS
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
    "results",
    exist_ok=True
)


tickets_df = pd.read_csv(
    DATA_PATH
)


required_columns = {
    "text",
    "category"
}


missing_columns = (
    required_columns
    - set(tickets_df.columns)
)


if missing_columns:
    raise ValueError(
        "Dataset is missing columns: "
        f"{missing_columns}"
    )


tickets_df = tickets_df[
    [
        "text",
        "category"
    ]
].dropna().copy()


tickets_df["text"] = (
    tickets_df["text"]
    .astype(str)
    .str.strip()
)


tickets_df["category"] = (
    tickets_df["category"]
    .astype(str)
    .str.strip()
    .str.lower()
)


tickets_df = tickets_df[
    tickets_df["text"].str.len() > 0
].reset_index(drop=True)


# Ensure every class has enough examples for splitting.
category_counts = (
    tickets_df["category"]
    .value_counts()
)


valid_categories = (
    category_counts[
        category_counts >= 5
    ]
    .index
)


tickets_df = tickets_df[
    tickets_df["category"].isin(
        valid_categories
    )
].reset_index(drop=True)


if tickets_df["category"].nunique() < 2:
    raise ValueError(
        "At least two ticket categories "
        "with five examples each are required."
    )


print("=" * 70)
print("BERT SUPPORT-TICKET ROUTER")
print("=" * 70)

print(
    "Ticket count:",
    len(tickets_df)
)

print(
    "\nCategory distribution:"
)

print(
    tickets_df["category"]
    .value_counts()
)


# ============================================================
# 3. ENCODE LABELS
# ============================================================

label_encoder = LabelEncoder()


tickets_df["label"] = (
    label_encoder.fit_transform(
        tickets_df["category"]
    )
)


class_names = list(
    label_encoder.classes_
)


number_of_classes = len(
    class_names
)


id_to_label = {
    index: class_name

    for index, class_name
    in enumerate(class_names)
}


label_to_id = {
    class_name: index

    for index, class_name
    in enumerate(class_names)
}


# ============================================================
# 4. SPLIT
# ============================================================

train_df, test_df = train_test_split(
    tickets_df,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=tickets_df["label"]
)


train_df = train_df.reset_index(
    drop=True
)

test_df = test_df.reset_index(
    drop=True
)


# ============================================================
# 5. TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


class TicketDataset(Dataset):

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

        encoded_ticket = self.tokenizer(
            self.texts[index],

            truncation=True,

            max_length=self.max_length
        )


        encoded_ticket["labels"] = int(
            self.labels[index]
        )


        return encoded_ticket


train_dataset = TicketDataset(
    train_df["text"],
    train_df["label"],
    tokenizer,
    MAX_LENGTH
)


test_dataset = TicketDataset(
    test_df["text"],
    test_df["label"],
    tokenizer,
    MAX_LENGTH
)


data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# 6. MODEL
# ============================================================

model = (
    AutoModelForSequenceClassification
    .from_pretrained(
        MODEL_NAME,

        num_labels=number_of_classes,

        id2label=id_to_label,

        label2id=label_to_id
    )
)


# ============================================================
# 7. MULTICLASS METRICS
# ============================================================

def calculate_metrics(evaluation_prediction):

    logits, labels = evaluation_prediction

    predictions = np.argmax(
        logits,
        axis=-1
    )


    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0
        )
    )


    return {
        "accuracy": accuracy_score(
            labels,
            predictions
        ),

        "precision": precision,

        "recall": recall,

        "f1": f1
    }


# ============================================================
# 8. TRAINER
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

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    save_total_limit=1,

    report_to="none",

    seed=RANDOM_STATE,

    fp16=torch.cuda.is_available()
)


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
# 9. TRAIN
# ============================================================

print(
    "\nTraining BERT ticket router..."
)


trainer.train()


# ============================================================
# 10. EVALUATE
# ============================================================

prediction_output = trainer.predict(
    test_dataset
)


ticket_predictions = np.argmax(
    prediction_output.predictions,
    axis=-1
)


ticket_labels = (
    test_df["label"]
    .to_numpy()
)


weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        ticket_labels,
        ticket_predictions,

        average="weighted",

        zero_division=0
    )
)


ticket_accuracy = accuracy_score(
    ticket_labels,
    ticket_predictions
)


print("\n" + "=" * 70)
print("TICKET ROUTING RESULTS")
print("=" * 70)

print(
    f"Accuracy : {ticket_accuracy:.4f}"
)

print(
    f"Precision: {weighted_precision:.4f}"
)

print(
    f"Recall   : {weighted_recall:.4f}"
)

print(
    f"F1-score : {weighted_f1:.4f}"
)


# ============================================================
# 11. PER-CLASS REPORT
# ============================================================

report = classification_report(
    ticket_labels,
    ticket_predictions,

    labels=list(
        range(number_of_classes)
    ),

    target_names=class_names,

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
    "\nPer-class report:"
)

print(
    classification_report(
        ticket_labels,
        ticket_predictions,

        labels=list(
            range(number_of_classes)
        ),

        target_names=class_names,

        zero_division=0
    )
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

ticket_confusion_matrix = (
    confusion_matrix(
        ticket_labels,
        ticket_predictions,

        labels=list(
            range(number_of_classes)
        )
    )
)


confusion_df = pd.DataFrame(
    ticket_confusion_matrix,

    index=[
        f"Actual {name}"
        for name in class_names
    ],

    columns=[
        f"Predicted {name}"
        for name in class_names
    ]
)


confusion_df.to_csv(
    CONFUSION_MATRIX_PATH
)


# ============================================================
# 13. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame([
    {
        "model": "BERT ticket router",

        "accuracy": round(
            ticket_accuracy,
            4
        ),

        "weighted_precision": round(
            weighted_precision,
            4
        ),

        "weighted_recall": round(
            weighted_recall,
            4
        ),

        "weighted_f1": round(
            weighted_f1,
            4
        )
    }
])


metrics_df.to_csv(
    METRICS_PATH,
    index=False
)


# ============================================================
# 14. SAVE MODEL AND LABELS
# ============================================================

trainer.save_model(
    MODEL_DIRECTORY
)


tokenizer.save_pretrained(
    MODEL_DIRECTORY
)


with open(
    LABELS_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            "id_to_label": id_to_label,
            "label_to_id": label_to_id
        },

        file,

        indent=4
    )


print("\nFiles saved:")

print(
    "Ticket router:",
    MODEL_DIRECTORY
)

print(
    "Metrics:",
    METRICS_PATH
)

print(
    "Report:",
    REPORT_PATH
)

print(
    "Confusion matrix:",
    CONFUSION_MATRIX_PATH
)