import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# 1. LOAD MNIST
# ============================================================

(
    X_train,
    y_train
), (
    X_test,
    y_test
) = keras.datasets.mnist.load_data()


# ============================================================
# 2. NORMALIZE IMAGES
# ============================================================

X_train = X_train.astype(
    "float32"
) / 255.0

X_test = X_test.astype(
    "float32"
) / 255.0


# ============================================================
# 3. BUILD MODEL
# ============================================================

model = keras.Sequential([
    layers.Input(
        shape=(28, 28)
    ),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(
        0.20
    ),

    layers.Dense(
        10,
        activation="softmax"
    )
])


# ============================================================
# 4. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",

    loss=(
        "sparse_categorical_crossentropy"
    ),

    metrics=["accuracy"]
)


# ============================================================
# 5. TRAIN
# ============================================================

model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=5,
    batch_size=32
)


# ============================================================
# 6. TEST
# ============================================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    "\nMNIST test accuracy:",
    round(test_accuracy, 4)
)