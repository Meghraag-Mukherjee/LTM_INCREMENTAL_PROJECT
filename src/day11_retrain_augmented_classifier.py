import csv
import json
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


ORIGINAL_MODEL_PATH = (
    "models/catalog_transfer.keras"
)

AUGMENTED_MODEL_PATH = (
    "models/catalog_augmented.keras"
)

CLASS_NAMES_PATH = (
    "models/catalog_class_names.json"
)

AUGMENTED_TRAIN_DIRECTORY = (
    "data/fashion_augmented_train"
)

VALIDATION_DIRECTORY = (
    "data/fashion_split/validation"
)

COMPARISON_PATH = (
    "results/augmentation_comparison.csv"
)

PER_CLASS_PATH = (
    "results/per_class_comparison.csv"
)

CONFUSION_MATRIX_PATH = (
    "results/augmented_confusion_matrix.csv"
)

IMAGE_SIZE = 224

BATCH_SIZE = 32

MAX_EPOCHS = 15

RANDOM_STATE = 42


tf.random.set_seed(RANDOM_STATE)


if not os.path.isdir(
    AUGMENTED_TRAIN_DIRECTORY
):
    raise FileNotFoundError(
        "Augmented training directory was not found. "
        "Run generate_synthetic_products.py first."
    )


if not os.path.isdir(
    VALIDATION_DIRECTORY
):
    raise FileNotFoundError(
        "Validation directory was not found."
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
# LOAD DATA
# ============================================================

train_dataset = (
    keras.utils.image_dataset_from_directory(
        AUGMENTED_TRAIN_DIRECTORY,

        image_size=(
            IMAGE_SIZE,
            IMAGE_SIZE
        ),

        batch_size=BATCH_SIZE,

        label_mode="int",

        shuffle=True,

        seed=RANDOM_STATE
    )
)


class_names = train_dataset.class_names

number_of_classes = len(
    class_names
)


validation_dataset = (
    keras.utils.image_dataset_from_directory(
        VALIDATION_DIRECTORY,

        class_names=class_names,

        image_size=(
            IMAGE_SIZE,
            IMAGE_SIZE
        ),

        batch_size=BATCH_SIZE,

        label_mode="int",

        shuffle=False
    )
)


train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

validation_dataset = (
    validation_dataset.prefetch(
        tf.data.AUTOTUNE
    )
)


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
# BUILD AUGMENTED MOBILE-NET MODEL
# ============================================================

augmentation = keras.Sequential([
    layers.RandomFlip(
        "horizontal"
    ),

    layers.RandomRotation(
        0.08
    ),

    layers.RandomZoom(
        0.10
    )
])


base_model = (
    keras.applications.MobileNetV2(
        weights="imagenet",

        include_top=False,

        input_shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        )
    )
)


base_model.trainable = False


inputs = keras.Input(
    shape=(
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    )
)

x = augmentation(
    inputs
)

x = (
    keras.applications.mobilenet_v2
    .preprocess_input(x)
)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(
    0.30
)(x)

outputs = layers.Dense(
    number_of_classes,
    activation="softmax"
)(x)

augmented_model = keras.Model(
    inputs,
    outputs,
    name="retailmind_augmented_classifier"
)


augmented_model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ModelCheckpoint(
        AUGMENTED_MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )
]


augmented_model.fit(
    train_dataset,

    validation_data=validation_dataset,

    epochs=MAX_EPOCHS,

    callbacks=callbacks,

    verbose=1
)


augmented_model = (
    keras.models.load_model(
        AUGMENTED_MODEL_PATH
    )
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    dataset
):

    loss, accuracy = model.evaluate(
        dataset,
        verbose=0
    )

    probabilities = model.predict(
        dataset,
        verbose=0
    )

    predictions = tf.argmax(
        probabilities,
        axis=1,
        output_type=tf.int32
    )

    actual_labels = tf.concat(
        [
            labels
            for images, labels
            in dataset
        ],
        axis=0
    )

    confusion_matrix = (
        tf.math.confusion_matrix(
            actual_labels,
            predictions,
            num_classes=number_of_classes
        )
    )

    per_class_accuracy = []

    for class_index in range(
        number_of_classes
    ):

        class_total = tf.reduce_sum(
            confusion_matrix[
                class_index
            ]
        )

        correct = confusion_matrix[
            class_index,
            class_index
        ]

        accuracy_value = tf.where(
            class_total > 0,

            tf.cast(
                correct,
                tf.float32
            )
            /
            tf.cast(
                class_total,
                tf.float32
            ),

            0.0
        )

        per_class_accuracy.append(
            float(
                accuracy_value.numpy()
            )
        )

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "confusion_matrix": (
            confusion_matrix.numpy()
        ),
        "per_class_accuracy": (
            per_class_accuracy
        )
    }


# ============================================================
# EVALUATE BEFORE AND AFTER
# ============================================================

augmented_results = evaluate_model(
    augmented_model,
    validation_dataset
)


original_results = None


if os.path.exists(
    ORIGINAL_MODEL_PATH
):

    try:

        original_model = (
            keras.models.load_model(
                ORIGINAL_MODEL_PATH
            )
        )

        original_results = evaluate_model(
            original_model,
            validation_dataset
        )

    except Exception as error:

        print(
            "\nOriginal Day-7 model could not "
            "be evaluated:"
        )

        print(error)


# ============================================================
# SAVE COMPARISON
# ============================================================

with open(
    COMPARISON_PATH,
    "w",
    encoding="utf-8",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "model",
        "validation_loss",
        "validation_accuracy"
    ])

    if original_results is not None:

        writer.writerow([
            "Day-7 MobileNetV2",
            round(
                original_results["loss"],
                4
            ),
            round(
                original_results["accuracy"],
                4
            )
        ])

    writer.writerow([
        "MobileNetV2 with VAE augmentation",
        round(
            augmented_results["loss"],
            4
        ),
        round(
            augmented_results["accuracy"],
            4
        )
    ])


with open(
    PER_CLASS_PATH,
    "w",
    encoding="utf-8",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "class_name",
        "before_accuracy",
        "after_accuracy",
        "accuracy_change"
    ])

    for class_index, class_name in enumerate(
        class_names
    ):

        after_accuracy = (
            augmented_results[
                "per_class_accuracy"
            ][class_index]
        )

        if original_results is not None:

            before_accuracy = (
                original_results[
                    "per_class_accuracy"
                ][class_index]
            )

            accuracy_change = (
                after_accuracy
                - before_accuracy
            )

        else:

            before_accuracy = ""
            accuracy_change = ""

        writer.writerow([
            class_name,

            (
                round(
                    before_accuracy,
                    4
                )
                if before_accuracy != ""
                else ""
            ),

            round(
                after_accuracy,
                4
            ),

            (
                round(
                    accuracy_change,
                    4
                )
                if accuracy_change != ""
                else ""
            )
        ])


with open(
    CONFUSION_MATRIX_PATH,
    "w",
    encoding="utf-8",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(
        ["actual/predicted"]
        + class_names
    )

    confusion = augmented_results[
        "confusion_matrix"
    ]

    for class_index, class_name in enumerate(
        class_names
    ):

        writer.writerow(
            [class_name]
            +
            confusion[
                class_index
            ].tolist()
        )


print("\n" + "=" * 70)
print("AUGMENTATION RESULTS")
print("=" * 70)


if original_results is not None:

    print(
        "Before accuracy:",
        round(
            original_results[
                "accuracy"
            ],
            4
        )
    )


print(
    "After accuracy :",
    round(
        augmented_results[
            "accuracy"
        ],
        4
    )
)


print("\nFiles saved:")

print(
    "Augmented model:",
    AUGMENTED_MODEL_PATH
)

print(
    "Overall comparison:",
    COMPARISON_PATH
)

print(
    "Per-class comparison:",
    PER_CLASS_PATH
)

print(
    "Confusion matrix:",
    CONFUSION_MATRIX_PATH
)