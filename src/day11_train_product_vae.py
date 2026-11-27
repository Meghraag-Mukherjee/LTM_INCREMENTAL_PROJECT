import csv
import json
import os
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


CONFIG_PATH = (
    "results/rare_class_config.json"
)

ENCODER_PATH = "models/vae_encoder.keras"

DECODER_PATH = "models/vae_decoder.keras"

HISTORY_PATH = (
    "results/vae_training_history.csv"
)

IMAGE_SIZE = 64

BATCH_SIZE = 32

LATENT_DIMENSION = 64

MAX_EPOCHS = 50

RANDOM_STATE = 42


random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(
        f"Configuration not found: {CONFIG_PATH}\n"
        "Run analyze_class_balance.py first."
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

training_directory = configuration[
    "training_directory"
]

rare_class_directory = os.path.join(
    training_directory,
    rare_class
)


if not os.path.isdir(rare_class_directory):
    raise FileNotFoundError(
        f"Rare-class directory not found: "
        f"{rare_class_directory}"
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
# LOAD RARE-CLASS DATA
# ============================================================

dataset = (
    keras.utils.image_dataset_from_directory(
        training_directory,

        labels="inferred",

        label_mode="int",

        class_names=[
            rare_class
        ],

        image_size=(
            IMAGE_SIZE,
            IMAGE_SIZE
        ),

        batch_size=BATCH_SIZE,

        shuffle=True,

        seed=RANDOM_STATE
    )
)


# Remove labels because VAE input and target are both images.
dataset = dataset.map(
    lambda images, labels: (
        tf.cast(images, tf.float32)
        / 255.0
    ),

    num_parallel_calls=(
        tf.data.AUTOTUNE
    )
)


dataset = dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# SAMPLING LAYER
# ============================================================

@keras.utils.register_keras_serializable()
class Sampling(layers.Layer):

    def call(self, inputs):

        z_mean, z_log_variance = inputs

        batch_size = tf.shape(
            z_mean
        )[0]

        latent_size = tf.shape(
            z_mean
        )[1]

        epsilon = tf.random.normal(
            shape=(
                batch_size,
                latent_size
            )
        )

        return (
            z_mean
            +
            tf.exp(
                0.5
                * z_log_variance
            )
            * epsilon
        )


# ============================================================
# ENCODER
# ============================================================

encoder_inputs = keras.Input(
    shape=(
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    ),

    name="product_image"
)

x = layers.Conv2D(
    32,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(encoder_inputs)

x = layers.Conv2D(
    64,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(x)

x = layers.Conv2D(
    128,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(x)

x = layers.Conv2D(
    256,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(x)

x = layers.Flatten()(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

z_mean = layers.Dense(
    LATENT_DIMENSION,
    name="z_mean"
)(x)

z_log_variance = layers.Dense(
    LATENT_DIMENSION,
    name="z_log_variance"
)(x)

z = Sampling(
    name="latent_sampling"
)([
    z_mean,
    z_log_variance
])

encoder = keras.Model(
    encoder_inputs,
    [
        z_mean,
        z_log_variance,
        z
    ],
    name="product_vae_encoder"
)


# ============================================================
# DECODER
# ============================================================

decoder_inputs = keras.Input(
    shape=(LATENT_DIMENSION,),
    name="latent_vector"
)

x = layers.Dense(
    4 * 4 * 256,
    activation="relu"
)(decoder_inputs)

x = layers.Reshape(
    (4, 4, 256)
)(x)

x = layers.Conv2DTranspose(
    256,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(x)

x = layers.Conv2DTranspose(
    128,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(x)

x = layers.Conv2DTranspose(
    64,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(x)

x = layers.Conv2DTranspose(
    32,
    3,
    strides=2,
    padding="same",
    activation="relu"
)(x)

decoder_outputs = layers.Conv2D(
    3,
    3,
    padding="same",
    activation="sigmoid",
    name="generated_product"
)(x)

decoder = keras.Model(
    decoder_inputs,
    decoder_outputs,
    name="product_vae_decoder"
)


# ============================================================
# VAE MODEL
# ============================================================

class ProductVAE(keras.Model):

    def __init__(
        self,
        encoder,
        decoder,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder

        self.total_loss_tracker = (
            keras.metrics.Mean(
                name="loss"
            )
        )

        self.reconstruction_loss_tracker = (
            keras.metrics.Mean(
                name="reconstruction_loss"
            )
        )

        self.kl_loss_tracker = (
            keras.metrics.Mean(
                name="kl_loss"
            )
        )

    @property
    def metrics(self):

        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    def train_step(self, images):

        with tf.GradientTape() as tape:

            (
                z_mean,
                z_log_variance,
                z
            ) = self.encoder(
                images,
                training=True
            )

            reconstructed_images = (
                self.decoder(
                    z,
                    training=True
                )
            )

            reconstruction_error = (
                tf.square(
                    images
                    - reconstructed_images
                )
            )

            reconstruction_loss = (
                tf.reduce_mean(
                    tf.reduce_sum(
                        reconstruction_error,
                        axis=(1, 2, 3)
                    )
                )
            )

            kl_loss = (
                -0.5
                * tf.reduce_mean(
                    tf.reduce_sum(
                        1
                        + z_log_variance
                        - tf.square(
                            z_mean
                        )
                        - tf.exp(
                            z_log_variance
                        ),
                        axis=1
                    )
                )
            )

            total_loss = (
                reconstruction_loss
                + kl_loss
            )

        gradients = tape.gradient(
            total_loss,
            self.trainable_weights
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.trainable_weights
            )
        )

        self.total_loss_tracker.update_state(
            total_loss
        )

        self.reconstruction_loss_tracker.update_state(
            reconstruction_loss
        )

        self.kl_loss_tracker.update_state(
            kl_loss
        )

        return {
            "loss": (
                self.total_loss_tracker.result()
            ),

            "reconstruction_loss": (
                self.reconstruction_loss_tracker.result()
            ),

            "kl_loss": (
                self.kl_loss_tracker.result()
            )
        }


vae = ProductVAE(
    encoder=encoder,
    decoder=decoder,
    name="product_vae"
)


vae.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0005
    )
)


print("=" * 70)
print("TRAINING PRODUCT VAE")
print("=" * 70)

print("Rare class:", rare_class)
print("Image size:", IMAGE_SIZE)
print("Latent dimension:", LATENT_DIMENSION)

encoder.summary()
decoder.summary()


history = vae.fit(
    dataset,
    epochs=MAX_EPOCHS,
    verbose=1
)


# ============================================================
# SAVE MODELS
# ============================================================

encoder.save(
    ENCODER_PATH
)

decoder.save(
    DECODER_PATH
)


history_keys = list(
    history.history.keys()
)


with open(
    HISTORY_PATH,
    "w",
    encoding="utf-8",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(
        ["epoch"] + history_keys
    )

    epoch_count = len(
        history.history[
            history_keys[0]
        ]
    )

    for epoch_index in range(
        epoch_count
    ):

        writer.writerow(
            [epoch_index + 1]
            +
            [
                history.history[key][
                    epoch_index
                ]
                for key in history_keys
            ]
        )


print("\nFiles saved:")

print("Encoder:", ENCODER_PATH)
print("Decoder:", DECODER_PATH)
print("History:", HISTORY_PATH)