import json
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

DATA_DIRECTORY = "data/fashion_products"

MODEL_PATH = "models/catalog_cnn.keras"

CLASS_NAMES_PATH = (
    "models/catalog_class_names.json"
)

HISTORY_PATH = (
    "results/catalog_training_history.csv"
)

CONFUSION_MATRIX_PATH = (
    "results/catalog_confusion_matrix.csv"
)

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

BATCH_SIZE = 32
VALIDATION_SPLIT = 0.20
MAX_EPOCHS = 30


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 3. CHECK DATASET
# ============================================================

if not os.path.isdir(DATA_DIRECTORY):
    raise FileNotFoundError(
        f"Dataset directory not found: "
        f"{DATA_DIRECTORY}\n"
        "Create one subfolder per product category."
    )

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# 4. LOAD TRAINING DATASET
# ============================================================

train_dataset = (
    keras.utils.image_dataset_from_directory(
        DATA_DIRECTORY,

        validation_split=VALIDATION_SPLIT,
        subset="training",

        seed=RANDOM_STATE,

        image_size=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH
        ),

        batch_size=BATCH_SIZE,

        label_mode="int"
    )
)


# ============================================================
# 5. LOAD VALIDATION DATASET
# ============================================================

validation_dataset = (
    keras.utils.image_dataset_from_directory(
        DATA_DIRECTORY,

        validation_split=VALIDATION_SPLIT,
        subset="validation",

        seed=RANDOM_STATE,

        image_size=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH
        ),

        batch_size=BATCH_SIZE,

        label_mode="int",

        # Keep validation ordering stable so labels match
        # predictions in the confusion matrix.
        shuffle=False
    )
)


# ============================================================
# 6. CLASS INFORMATION
# ============================================================

class_names = train_dataset.class_names

number_of_classes = len(class_names)

print("=" * 70)
print("RETAILMIND CATALOG CNN")
print("=" * 70)

print("Classes:", class_names)
print("Number of classes:", number_of_classes)

if number_of_classes < 2:
    raise ValueError(
        "At least two class folders are required."
    )


# ============================================================
# 7. SAVE CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        class_names,
        file,
        indent=4
    )


# ============================================================
# 8. OPTIMIZE DATA PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = (
    validation_dataset.prefetch(
        buffer_size=AUTOTUNE
    )
)


# ============================================================
# 9. DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip(
            mode="horizontal"
        ),

        layers.RandomRotation(
            factor=0.08
        ),

        layers.RandomZoom(
            height_factor=0.10,
            width_factor=0.10
        ),

        layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        )
    ],
    name="data_augmentation"
)


# ============================================================
# 10. BUILD CNN
# ============================================================

model = keras.Sequential(
    [
        layers.Input(
            shape=(
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
                3
            ),
            name="product_image"
        ),

        data_augmentation,

        layers.Rescaling(
            scale=1.0 / 255.0,
            name="normalize_pixels"
        ),

        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            name="convolution_1"
        ),

        layers.BatchNormalization(
            name="batch_normalization_1"
        ),

        layers.MaxPooling2D(
            pool_size=(2, 2),
            name="pooling_1"
        ),

        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            name="convolution_2"
        ),

        layers.BatchNormalization(
            name="batch_normalization_2"
        ),

        layers.MaxPooling2D(
            pool_size=(2, 2),
            name="pooling_2"
        ),

        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            name="convolution_3"
        ),

        layers.BatchNormalization(
            name="batch_normalization_3"
        ),

        layers.MaxPooling2D(
            pool_size=(2, 2),
            name="pooling_3"
        ),

        layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            name="convolution_4"
        ),

        layers.MaxPooling2D(
            pool_size=(2, 2),
            name="pooling_4"
        ),

        layers.Dropout(
            rate=0.30,
            name="feature_dropout"
        ),

        layers.GlobalAveragePooling2D(
            name="global_average_pooling"
        ),

        layers.Dense(
            units=256,
            activation="relu",
            name="dense_layer"
        ),

        layers.BatchNormalization(
            name="dense_batch_normalization"
        ),

        layers.Dropout(
            rate=0.50,
            name="dense_dropout"
        ),

        layers.Dense(
            units=number_of_classes,
            activation="softmax",
            name="category_probabilities"
        )
    ],
    name="retailmind_catalog_cnn"
)


# ============================================================
# 11. COMPILE
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )
    ]
)

print("\nCatalog CNN architecture:")
model.summary()


# ============================================================
# 12. CALLBACKS
# ============================================================

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    mode="min",
    restore_best_weights=True,
    verbose=1
)

reduce_learning_rate = (
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001,
        verbose=1
    )
)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)


# ============================================================
# 13. TRAIN
# ============================================================

history = model.fit(
    train_dataset,

    validation_data=validation_dataset,

    epochs=MAX_EPOCHS,

    callbacks=[
        early_stopping,
        reduce_learning_rate,
        model_checkpoint
    ],

    verbose=1
)


# ============================================================
# 14. LOAD THE BEST SAVED MODEL
# ============================================================

best_model = keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# 15. VALIDATION RESULTS
# ============================================================

validation_loss, validation_accuracy = (
    best_model.evaluate(
        validation_dataset,
        verbose=0
    )
)

print("\n" + "=" * 70)
print("CATALOG CNN VALIDATION RESULTS")
print("=" * 70)

print(
    f"Validation loss    : "
    f"{validation_loss:.4f}"
)

print(
    f"Validation accuracy: "
    f"{validation_accuracy:.4f}"
)


# ============================================================
# 16. VALIDATION PREDICTIONS
# ============================================================

probabilities = best_model.predict(
    validation_dataset,
    verbose=0
)

predictions = np.argmax(
    probabilities,
    axis=1
)

true_labels = np.concatenate([
    labels.numpy()
    for _, labels in validation_dataset
])


# ============================================================
# 17. CONFUSION MATRIX
# ============================================================

confusion_matrix = tf.math.confusion_matrix(
    labels=true_labels,
    predictions=predictions,
    num_classes=number_of_classes
).numpy()

confusion_matrix_df = pd.DataFrame(
    confusion_matrix,
    index=class_names,
    columns=class_names
)

print("\nConfusion matrix:")
print(confusion_matrix_df)


# ============================================================
# 18. PER-CLASS ACCURACY
# ============================================================

print("\nPer-class validation accuracy:")

for class_index, class_name in enumerate(
    class_names
):
    class_total = confusion_matrix[
        class_index
    ].sum()

    correct = confusion_matrix[
        class_index,
        class_index
    ]

    class_accuracy = (
        correct / class_total
        if class_total > 0
        else 0
    )

    print(
        f"{class_name:<25}: "
        f"{class_accuracy:.4f}"
    )


# ============================================================
# 19. SAVE HISTORY AND CONFUSION MATRIX
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df["final_validation_accuracy"] = np.nan

history_df.loc[
    history_df.index[-1],
    "final_validation_accuracy"
] = validation_accuracy

history_df.to_csv(
    HISTORY_PATH,
    index=False
)

confusion_matrix_df.to_csv(
    CONFUSION_MATRIX_PATH
)

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print("Catalog model    :", MODEL_PATH)
print("Class names      :", CLASS_NAMES_PATH)
print("Training history :", HISTORY_PATH)
print("Confusion matrix :", CONFUSION_MATRIX_PATH)