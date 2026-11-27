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

MODEL_PATH = "models/fashion_mnist_cnn.keras"

HISTORY_PATH = "results/fashion_mnist_history.csv"

CONFUSION_MATRIX_PATH = (
    "results/fashion_mnist_confusion_matrix.csv"
)

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# 3. OUTPUT DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# 4. LOAD FASHION-MNIST
# ============================================================

(
    X_train,
    y_train
), (
    X_test,
    y_test
) = keras.datasets.fashion_mnist.load_data()

print("=" * 70)
print("FASHION-MNIST CNN PROTOTYPE")
print("=" * 70)

print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)
print("Testing images :", X_test.shape)
print("Testing labels :", y_test.shape)


# ============================================================
# 5. NORMALIZE AND RESHAPE
# ============================================================

# Original pixel values are between 0 and 255.
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Add the channel dimension.
# Before: (60000, 28, 28)
# After:  (60000, 28, 28, 1)
X_train = np.expand_dims(
    X_train,
    axis=-1
)

X_test = np.expand_dims(
    X_test,
    axis=-1
)

print("\nCNN training input:", X_train.shape)
print("CNN testing input :", X_test.shape)


# ============================================================
# 6. BUILD CNN
# ============================================================

model = keras.Sequential(
    [
        layers.Input(
            shape=(28, 28, 1),
            name="fashion_image"
        ),

        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
            name="convolution_1"
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

        layers.MaxPooling2D(
            pool_size=(2, 2),
            name="pooling_3"
        ),

        layers.Dropout(
            rate=0.25,
            name="convolution_dropout"
        ),

        layers.Flatten(
            name="flatten_features"
        ),

        layers.Dense(
            units=128,
            activation="relu",
            name="dense_layer"
        ),

        layers.Dropout(
            rate=0.50,
            name="dense_dropout"
        ),

        layers.Dense(
            units=10,
            activation="softmax",
            name="category_probabilities"
        )
    ],
    name="fashion_mnist_cnn"
)


# ============================================================
# 7. COMPILE
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

print("\nCNN architecture:")

model.summary()


# ============================================================
# 8. CALLBACKS
# ============================================================

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
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


# ============================================================
# 9. TRAIN
# ============================================================

history = model.fit(
    X_train,
    y_train,

    validation_split=0.20,

    epochs=20,
    batch_size=64,

    callbacks=[
        early_stopping,
        reduce_learning_rate
    ],

    verbose=1
)


# ============================================================
# 10. EVALUATE
# ============================================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n" + "=" * 70)
print("FASHION-MNIST TEST RESULTS")
print("=" * 70)

print(f"Test loss    : {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")


# ============================================================
# 11. PREDICTIONS
# ============================================================

probabilities = model.predict(
    X_test,
    batch_size=128,
    verbose=0
)

predictions = np.argmax(
    probabilities,
    axis=1
)


# ============================================================
# 12. CONFUSION MATRIX USING TENSORFLOW
# ============================================================

confusion_matrix = tf.math.confusion_matrix(
    labels=y_test,
    predictions=predictions,
    num_classes=10
).numpy()

confusion_matrix_df = pd.DataFrame(
    confusion_matrix,
    index=CLASS_NAMES,
    columns=CLASS_NAMES
)

print("\nConfusion matrix:")
print(confusion_matrix_df)


# ============================================================
# 13. PER-CLASS ACCURACY
# ============================================================

print("\nPer-class accuracy:")

for class_index, class_name in enumerate(
    CLASS_NAMES
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
        f"{class_name:<15}: "
        f"{class_accuracy:.4f}"
    )


# ============================================================
# 14. SAVE RESULTS
# ============================================================

model.save(MODEL_PATH)

history_df = pd.DataFrame(
    history.history
)

history_df["test_accuracy"] = np.nan
history_df.loc[
    history_df.index[-1],
    "test_accuracy"
] = test_accuracy

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

print("CNN model       :", MODEL_PATH)
print("Training history:", HISTORY_PATH)
print("Confusion matrix:", CONFUSION_MATRIX_PATH)