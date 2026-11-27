
import argparse
import json
import os

import numpy as np

from tensorflow import keras


# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_PATH = "models/catalog_cnn.keras"

CLASS_NAMES_PATH = (
    "models/catalog_class_names.json"
)

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128


# ============================================================
# 2. LOAD MODEL AND CLASS NAMES
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Run python -m src.train_catalog_cnn first."
    )

if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(
        f"Class-name file not found: "
        f"{CLASS_NAMES_PATH}"
    )

model = keras.models.load_model(
    MODEL_PATH
)

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:
    class_names = json.load(file)


# ============================================================
# 3. PREDICTION FUNCTION
# ============================================================

def predict_product(
    image_path,
    top_k=3
):
    """
    Predict the category of one product image.

    Parameters
    ----------
    image_path : str
        Path to the product image.

    top_k : int
        Number of highest-probability categories returned.

    Returns
    -------
    dict
        Predicted category, confidence and alternatives.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = keras.utils.load_img(
        image_path,
        target_size=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH
        )
    )

    image_array = keras.utils.img_to_array(
        image
    )

    # Add batch dimension:
    # (128, 128, 3) -> (1, 128, 128, 3)
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    top_k = min(
        top_k,
        len(class_names)
    )

    top_indices = np.argsort(
        probabilities
    )[-top_k:][::-1]

    top_predictions = [
        {
            "category": class_names[
                int(index)
            ],
            "probability": round(
                float(probabilities[index]),
                4
            )
        }
        for index in top_indices
    ]

    best_index = int(
        top_indices[0]
    )

    return {
        "predicted_category": (
            class_names[best_index]
        ),

        "confidence": round(
            float(probabilities[best_index]),
            4
        ),

        "top_predictions": top_predictions
    }


# ============================================================
# 4. COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Predict a product category using "
            "the RetailMind catalog CNN."
        )
    )

    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the product image."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of predictions to return."
    )

    arguments = parser.parse_args()

    result = predict_product(
        arguments.image_path,
        top_k=arguments.top_k
    )

    print("\nCatalog prediction:")

    print(
        "Predicted category:",
        result["predicted_category"]
    )

    print(
        "Confidence:",
        result["confidence"]
    )

    print("\nTop predictions:")

    for prediction in result[
        "top_predictions"
    ]:
        print(
            f"- {prediction['category']}: "
            f"{prediction['probability']}"
        )