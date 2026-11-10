import json
import os

import numpy as np

from tensorflow import keras


MODEL_PATH = (
    "models/catalog_resnet_finetuned.keras"
)

CLASS_NAMES_PATH = (
    "models/catalog_class_names.json"
)

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224


class ProductClassifier:
    """
    RetailMind product-image classifier.
    """

    def __init__(
        self,
        model_path=MODEL_PATH,
        class_names_path=CLASS_NAMES_PATH
    ):

        if not os.path.exists(
            model_path
        ):
            raise FileNotFoundError(
                f"Classifier not found: {model_path}\n"
                "Run python -m vision.train_transfer first."
            )

        if not os.path.exists(
            class_names_path
        ):
            raise FileNotFoundError(
                f"Class-name file not found: "
                f"{class_names_path}"
            )

        self.model = keras.models.load_model(
            model_path
        )

        with open(
            class_names_path,
            "r",
            encoding="utf-8"
        ) as file:
            self.class_names = json.load(file)

    def predict(
        self,
        image_path,
        top_k=3
    ):
        """
        Classify a single product image.
        """

        if not os.path.exists(
            image_path
        ):
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

        image_array = (
            keras.utils.img_to_array(
                image
            )
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # Do not preprocess manually.
        # ResNet preprocessing is already stored
        # inside the saved model.
        probabilities = self.model.predict(
            image_array,
            verbose=0
        )[0]

        top_k = min(
            top_k,
            len(self.class_names)
        )

        top_indices = np.argsort(
            probabilities
        )[-top_k:][::-1]

        predictions = []

        for index in top_indices:
            predictions.append({
                "category": (
                    self.class_names[
                        int(index)
                    ]
                ),

                "probability": round(
                    float(
                        probabilities[index]
                    ),
                    4
                )
            })

        best_prediction = predictions[0]

        return {
            "predicted_category": (
                best_prediction["category"]
            ),

            "confidence": (
                best_prediction["probability"]
            ),

            "top_predictions": predictions
        }