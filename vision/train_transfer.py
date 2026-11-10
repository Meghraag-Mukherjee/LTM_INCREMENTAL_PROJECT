import json
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 1. SETTINGS
# ============================================================

RANDOM_STATE = 42

DATA_DIRECTORY = "data/fashion_products"

OLD_CNN_PATH = "models/catalog_cnn.keras"

FROZEN_MODEL_PATH = (
    "models/catalog_resnet_frozen.keras"
)

FINE_TUNED_MODEL_PATH = (
    "models/catalog_resnet_finetuned.keras"
)

CLASS_NAMES_PATH = (
    "models/catalog_class_names.json"
)

FROZEN_HISTORY_PATH = (
    "results/transfer_learning_history.csv"
)

FINE_TUNING_HISTORY_PATH = (
    "results/fine_tuning_history.csv"
)

CONFUSION_MATRIX_PATH = (
    "results/transfer_confusion_matrix.csv"
)

COMPARISON_PATH = (
    "results/vision_model_comparison.csv"
)

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

BATCH_SIZE = 32

VALIDATION_SPLIT = 0.20

FROZEN_EPOCHS = 15

FINE_TUNE_EPOCHS = 10

FINE_TUNE_LAYER_COUNT = 30


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 3. CHECK DIRECTORIES
# ============================================================

if not os.path.isdir(DATA_DIRECTORY):
    raise FileNotFoundError(
        f"Dataset directory was not found: "
        f"{DATA_DIRECTORY}\n"
        "Create one folder per product category."
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
# 4. LOAD TRAINING DATA
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
# 5. LOAD VALIDATION DATA
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

        shuffle=False
    )
)


# ============================================================
# 6. CLASS INFORMATION
# ============================================================

class_names = train_dataset.class_names

number_of_classes = len(class_names)

if number_of_classes < 2:
    raise ValueError(
        "At least two category folders are required."
    )

print("=" * 70)
print("RETAILMIND TRANSFER LEARNING")
print("=" * 70)

print("Product classes:")

for index, class_name in enumerate(
    class_names
):
    print(index, class_name)

print(
    "\nNumber of classes:",
    number_of_classes
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
# 8. OPTIMIZE INPUT PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    AUTOTUNE
)

validation_dataset = (
    validation_dataset.prefetch(
        AUTOTUNE
    )
)


# ============================================================
# 9. DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.08
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        )
    ],

    name="data_augmentation"
)


# ============================================================
# 10. LOAD PRETRAINED RESNET50
# ============================================================

base_model = (
    keras.applications.ResNet50(
        weights="imagenet",

        include_top=False,

        input_shape=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            3
        )
    )
)

# Freeze the ImageNet backbone initially.
base_model.trainable = False


# ============================================================
# 11. BUILD TRANSFER-LEARNING MODEL
# ============================================================

inputs = keras.Input(
    shape=(
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        3
    ),

    name="product_image"
)

x = data_augmentation(
    inputs
)

# Required preprocessing for ResNet50.
x = keras.applications.resnet50.preprocess_input(
    x
)

# training=False keeps BatchNormalization layers
# in inference mode while the backbone is frozen.
x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)

x = layers.BatchNormalization(
    name="head_batch_normalization"
)(x)

x = layers.Dropout(
    0.40,
    name="head_dropout_1"
)(x)

x = layers.Dense(
    256,
    activation="relu",
    name="head_dense"
)(x)

x = layers.Dropout(
    0.30,
    name="head_dropout_2"
)(x)

outputs = layers.Dense(
    number_of_classes,
    activation="softmax",
    name="category_probabilities"
)(x)

model = keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="retailmind_resnet50"
)


# ============================================================
# 12. COMPILE FROZEN MODEL
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

print("\nFrozen transfer-learning model:")

model.summary()


# ============================================================
# 13. CALLBACKS FOR FROZEN TRAINING
# ============================================================

frozen_early_stopping = (
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        mode="min",
        restore_best_weights=True,
        verbose=1
    )
)

frozen_reduce_lr = (
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001,
        verbose=1
    )
)

frozen_checkpoint = (
    keras.callbacks.ModelCheckpoint(
        filepath=FROZEN_MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )
)


# ============================================================
# 14. TRAIN NEW CLASSIFICATION HEAD
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1: TRAINING THE NEW CLASSIFICATION HEAD")
print("=" * 70)

frozen_history = model.fit(
    train_dataset,

    validation_data=validation_dataset,

    epochs=FROZEN_EPOCHS,

    callbacks=[
        frozen_early_stopping,
        frozen_reduce_lr,
        frozen_checkpoint
    ],

    verbose=1
)


# ============================================================
# 15. EVALUATE FROZEN MODEL
# ============================================================

best_frozen_model = (
    keras.models.load_model(
        FROZEN_MODEL_PATH
    )
)

frozen_loss, frozen_accuracy = (
    best_frozen_model.evaluate(
        validation_dataset,
        verbose=0
    )
)

print("\nFrozen model results:")

print(
    f"Validation loss    : "
    f"{frozen_loss:.4f}"
)

print(
    f"Validation accuracy: "
    f"{frozen_accuracy:.4f}"
)


# ============================================================
# 16. PREPARE FOR FINE-TUNING
# ============================================================

# Continue using the current model object and its trained head.
base_model.trainable = True

# Freeze all except the final layers.
for layer in base_model.layers[
    :-FINE_TUNE_LAYER_COUNT
]:
    layer.trainable = False

# Keep BatchNormalization layers frozen.
# This helps avoid destabilizing their stored statistics
# on relatively small product datasets.
for layer in base_model.layers:
    if isinstance(
        layer,
        layers.BatchNormalization
    ):
        layer.trainable = False

trainable_backbone_layers = sum(
    int(layer.trainable)
    for layer in base_model.layers
)

print(
    "\nTrainable ResNet layers:",
    trainable_backbone_layers
)


# ============================================================
# 17. RECOMPILE BEFORE FINE-TUNING
# ============================================================

# Recompilation is required after changing trainable layers.
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.00001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        )
    ]
)


# ============================================================
# 18. FINE-TUNING CALLBACKS
# ============================================================

fine_tune_early_stopping = (
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        mode="min",
        restore_best_weights=True,
        verbose=1
    )
)

fine_tune_reduce_lr = (
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.000001,
        verbose=1
    )
)

fine_tune_checkpoint = (
    keras.callbacks.ModelCheckpoint(
        filepath=FINE_TUNED_MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )
)


# ============================================================
# 19. FINE-TUNE
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2: FINE-TUNING TOP RESNET LAYERS")
print("=" * 70)

fine_tuning_history = model.fit(
    train_dataset,

    validation_data=validation_dataset,

    epochs=FINE_TUNE_EPOCHS,

    callbacks=[
        fine_tune_early_stopping,
        fine_tune_reduce_lr,
        fine_tune_checkpoint
    ],

    verbose=1
)


# ============================================================
# 20. LOAD AND EVALUATE FINE-TUNED MODEL
# ============================================================

best_fine_tuned_model = (
    keras.models.load_model(
        FINE_TUNED_MODEL_PATH
    )
)

fine_tuned_loss, fine_tuned_accuracy = (
    best_fine_tuned_model.evaluate(
        validation_dataset,
        verbose=0
    )
)

print("\nFine-tuned model results:")

print(
    f"Validation loss    : "
    f"{fine_tuned_loss:.4f}"
)

print(
    f"Validation accuracy: "
    f"{fine_tuned_accuracy:.4f}"
)


# ============================================================
# 21. EVALUATE PREVIOUS CNN
# ============================================================

comparison_rows = [
    {
        "Model": "ResNet50 frozen backbone",
        "Validation Loss": round(
            frozen_loss,
            4
        ),
        "Validation Accuracy": round(
            frozen_accuracy,
            4
        )
    },

    {
        "Model": "ResNet50 fine-tuned",
        "Validation Loss": round(
            fine_tuned_loss,
            4
        ),
        "Validation Accuracy": round(
            fine_tuned_accuracy,
            4
        )
    }
]

if os.path.exists(
    OLD_CNN_PATH
):
    try:
        old_cnn = keras.models.load_model(
            OLD_CNN_PATH
        )

        old_loss, old_accuracy = (
            old_cnn.evaluate(
                validation_dataset,
                verbose=0
            )
        )

        comparison_rows.append({
            "Model": "Day-6 CNN from scratch",

            "Validation Loss": round(
                old_loss,
                4
            ),

            "Validation Accuracy": round(
                old_accuracy,
                4
            )
        })

    except Exception as error:
        print(
            "\nPrevious CNN could not be evaluated "
            "on 224 x 224 data."
        )

        print(
            "Reason:",
            error
        )

        print(
            "Use the saved Day-6 validation accuracy "
            "from the previous training run."
        )

else:
    print(
        "\nPrevious model was not found:",
        OLD_CNN_PATH
    )


# ============================================================
# 22. MODEL COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df = comparison_df.sort_values(
    by="Validation Accuracy",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 80)
print("VISION MODEL COMPARISON")
print("=" * 80)

print(
    comparison_df.to_string(
        index=False
    )
)

print(
    "\nCurrent winner:",
    comparison_df.iloc[0]["Model"]
)


# ============================================================
# 23. CONFUSION MATRIX
# ============================================================

probabilities = (
    best_fine_tuned_model.predict(
        validation_dataset,
        verbose=0
    )
)

predictions = np.argmax(
    probabilities,
    axis=1
)

true_labels = np.concatenate([
    labels.numpy()
    for _, labels in validation_dataset
])

confusion = tf.math.confusion_matrix(
    labels=true_labels,
    predictions=predictions,
    num_classes=number_of_classes
).numpy()

confusion_df = pd.DataFrame(
    confusion,
    index=class_names,
    columns=class_names
)

print("\nFine-tuned model confusion matrix:")

print(confusion_df)


# ============================================================
# 24. PER-CLASS ACCURACY
# ============================================================

print("\nPer-class validation accuracy:")

for class_index, class_name in enumerate(
    class_names
):
    total = confusion[
        class_index
    ].sum()

    correct = confusion[
        class_index,
        class_index
    ]

    class_accuracy = (
        correct / total
        if total > 0
        else 0
    )

    print(
        f"{class_name:<25}: "
        f"{class_accuracy:.4f}"
    )


# ============================================================
# 25. SAVE RESULTS
# ============================================================

pd.DataFrame(
    frozen_history.history
).to_csv(
    FROZEN_HISTORY_PATH,
    index=False
)

pd.DataFrame(
    fine_tuning_history.history
).to_csv(
    FINE_TUNING_HISTORY_PATH,
    index=False
)

comparison_df.to_csv(
    COMPARISON_PATH,
    index=False
)

confusion_df.to_csv(
    CONFUSION_MATRIX_PATH
)

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    "Frozen model:",
    FROZEN_MODEL_PATH
)

print(
    "Fine-tuned model:",
    FINE_TUNED_MODEL_PATH
)

print(
    "Class names:",
    CLASS_NAMES_PATH
)

print(
    "Model comparison:",
    COMPARISON_PATH
)

print(
    "Confusion matrix:",
    CONFUSION_MATRIX_PATH
)
