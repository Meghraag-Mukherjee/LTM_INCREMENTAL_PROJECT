import os
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import matplotlib.cm as cm


MODEL_PATH = "models/catalog_transfer.keras"
CLASS_NAMES_PATH = "models/catalog_class_names.json"

IMAGE_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

model = keras.models.load_model(MODEL_PATH)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:
    class_names = json.load(file)


# ============================================================
# FIND CONVOLUTIONAL BACKBONE
# ============================================================

def find_backbone(model):
    """
    Find the nested transfer-learning backbone.

    Day 7 normally contains MobileNetV2 inside the
    complete classification model.
    """

    for layer in model.layers:

        if isinstance(layer, keras.Model):

            # MobileNetV2 contains Conv2D layers
            has_conv = any(
                isinstance(
                    sublayer,
                    keras.layers.Conv2D
                )
                for sublayer in layer.layers
            )

            if has_conv:
                return layer

    raise ValueError(
        "Could not find a convolutional backbone "
        "inside the saved model."
    )


backbone = find_backbone(model)

print(
    "Backbone:",
    backbone.name
)


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_last_conv_layer(backbone):

    for layer in reversed(backbone.layers):

        output_shape = getattr(
            layer.output,
            "shape",
            None
        )

        if (
            output_shape is not None
            and len(output_shape) == 4
        ):
            return layer.name

    raise ValueError(
        "No 4D convolutional feature layer found."
    )


LAST_CONV_LAYER = find_last_conv_layer(
    backbone
)

print(
    "Last convolutional feature layer:",
    LAST_CONV_LAYER
)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):

    image = keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = (
        keras.utils.img_to_array(image)
    )

    image_batch = np.expand_dims(
        image_array,
        axis=0
    )

    return image, image_batch


# ============================================================
# CREATE GRAD-CAM HEATMAP
# ============================================================

def make_gradcam_heatmap(
    image_batch,
    model,
    backbone,
    last_conv_layer_name,
    class_index=None
):

    last_conv_layer = backbone.get_layer(
        last_conv_layer_name
    )

    # Model from backbone input to:
    # 1. requested convolutional output
    # 2. final backbone output
    backbone_grad_model = keras.Model(
        backbone.inputs,
        [
            last_conv_layer.output,
            backbone.output
        ]
    )

    # Locate backbone position in complete model
    backbone_index = model.layers.index(
        backbone
    )

    with tf.GradientTape() as tape:

        # Run layers before backbone
        x = tf.cast(
            image_batch,
            tf.float32
        )

        for layer in model.layers[
            1:backbone_index
        ]:
            x = layer(
                x,
                training=False
            )

        conv_output, x = (
            backbone_grad_model(
                x,
                training=False
            )
        )

        # Run classification head
        for layer in model.layers[
            backbone_index + 1:
        ]:
            x = layer(
                x,
                training=False
            )

        predictions = x

        if class_index is None:
            class_index = tf.argmax(
                predictions[0]
            )

        class_score = predictions[
            :,
            class_index
        ]

    gradients = tape.gradient(
        class_score,
        conv_output
    )

    # Importance of each feature map
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_output = conv_output[0]

    heatmap = tf.reduce_sum(
        conv_output
        * pooled_gradients,
        axis=-1
    )

    # Keep only positive influence
    heatmap = tf.maximum(
        heatmap,
        0
    )

    maximum = tf.reduce_max(
        heatmap
    )

    heatmap = tf.where(
        maximum > 0,
        heatmap / maximum,
        heatmap
    )

    return (
        heatmap.numpy(),
        predictions.numpy()[0],
        int(class_index)
    )


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(
    original_image,
    heatmap,
    alpha=0.40
):

    original_array = (
        keras.utils.img_to_array(
            original_image
        )
    )

    # Convert 0-1 heatmap into 0-255
    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    jet = cm.get_cmap("jet")

    jet_colors = jet(
        np.arange(256)
    )[:, :3]

    colored_heatmap = (
        jet_colors[
            heatmap_uint8
        ]
    )

    colored_heatmap = (
        keras.utils.array_to_img(
            colored_heatmap
        )
    )

    colored_heatmap = colored_heatmap.resize(
        original_image.size
    )

    colored_heatmap = (
        keras.utils.img_to_array(
            colored_heatmap
        )
    )

    overlay = (
        original_array * (1 - alpha)
        +
        colored_heatmap * alpha
    )

    overlay = np.clip(
        overlay,
        0,
        255
    )

    return overlay
