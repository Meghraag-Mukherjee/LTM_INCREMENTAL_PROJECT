import json
import os
import shutil

import tensorflow as tf
from tensorflow import keras


CONFIG_PATH = (
    "results/rare_class_config.json"
)

DECODER_PATH = "models/vae_decoder.keras"

ORIGINAL_TRAIN_DIRECTORY = (
    "data/fashion_split/train"
)

AUGMENTED_TRAIN_DIRECTORY = (
    "data/fashion_augmented_train"
)

SYNTHETIC_RESULT_DIRECTORY = (
    "results/synthetic_products"
)

LATENT_DIMENSION = 64

MAX_SYNTHETIC_IMAGES = 1000

RANDOM_STATE = 42


tf.random.set_seed(RANDOM_STATE)


if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(
        f"Configuration not found: {CONFIG_PATH}"
    )


if not os.path.exists(DECODER_PATH):
    raise FileNotFoundError(
        f"Decoder not found: {DECODER_PATH}\n"
        "Run train_product_vae.py first."
    )


with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as file:

    configuration = json.load(file)


rare_class = configuration[
    "rare_class"
]

requested_count = int(
    configuration[
        "required_synthetic_count"
    ]
)

# Avoid generating an excessive number in one lab run.
synthetic_count = min(
    requested_count,
    MAX_SYNTHETIC_IMAGES
)


if synthetic_count <= 0:
    synthetic_count = 100


# Create a fresh augmented training dataset.
if os.path.exists(
    AUGMENTED_TRAIN_DIRECTORY
):
    shutil.rmtree(
        AUGMENTED_TRAIN_DIRECTORY
    )


shutil.copytree(
    ORIGINAL_TRAIN_DIRECTORY,
    AUGMENTED_TRAIN_DIRECTORY
)


augmented_rare_directory = os.path.join(
    AUGMENTED_TRAIN_DIRECTORY,
    rare_class
)


synthetic_result_class_directory = os.path.join(
    SYNTHETIC_RESULT_DIRECTORY,
    rare_class
)


os.makedirs(
    augmented_rare_directory,
    exist_ok=True
)

os.makedirs(
    synthetic_result_class_directory,
    exist_ok=True
)


decoder = keras.models.load_model(
    DECODER_PATH
)


BATCH_SIZE = 32

generated_count = 0


print("=" * 70)
print("GENERATING SYNTHETIC PRODUCT IMAGES")
print("=" * 70)

print("Rare class:", rare_class)
print("Images to create:", synthetic_count)


while generated_count < synthetic_count:

    current_batch_size = min(
        BATCH_SIZE,
        synthetic_count
        - generated_count
    )

    latent_vectors = tf.random.normal(
        shape=(
            current_batch_size,
            LATENT_DIMENSION
        )
    )

    generated_images = decoder.predict(
        latent_vectors,
        verbose=0
    )

    for batch_index in range(
        current_batch_size
    ):

        image_number = (
            generated_count
            + batch_index
            + 1
        )

        image = generated_images[
            batch_index
        ]

        image = tf.clip_by_value(
            image,
            0.0,
            1.0
        )

        image = tf.cast(
            image * 255.0,
            tf.uint8
        )

        encoded_image = tf.io.encode_png(
            image
        )

        file_name = (
            f"synthetic_{rare_class}_"
            f"{image_number:05d}.png"
        )

        augmented_path = os.path.join(
            augmented_rare_directory,
            file_name
        )

        result_path = os.path.join(
            synthetic_result_class_directory,
            file_name
        )

        tf.io.write_file(
            augmented_path,
            encoded_image
        )

        tf.io.write_file(
            result_path,
            encoded_image
        )

    generated_count += (
        current_batch_size
    )

    print(
        f"Generated "
        f"{generated_count}/"
        f"{synthetic_count}"
    )


print("\nSynthetic dataset created:")

print(AUGMENTED_TRAIN_DIRECTORY)

print("\nGenerated images saved:")

print(
    synthetic_result_class_directory
)