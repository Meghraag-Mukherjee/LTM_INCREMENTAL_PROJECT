import argparse
import json
import os

from vision.classifier import ProductClassifier
from vision.detector import ProductDetector


class RetailMindVision:
    """
    Combined RetailMind vision module.

    Classification:
        Determines the main product category.

    Detection:
        Finds objects and their bounding boxes.
    """

    def __init__(
        self,
        classifier_model=(
            "models/"
            "catalog_resnet_finetuned.keras"
        ),
        class_names_file=(
            "models/"
            "catalog_class_names.json"
        ),
        detector_model="yolo11n.pt",
        detection_threshold=0.25
    ):

        self.classifier = ProductClassifier(
            model_path=classifier_model,

            class_names_path=(
                class_names_file
            )
        )

        self.detector = ProductDetector(
            model_path=detector_model,

            confidence_threshold=(
                detection_threshold
            )
        )

    def analyze(
        self,
        image_path,
        top_k=3,
        save_detection=True
    ):
        """
        Run classification and detection.
        """

        if not os.path.exists(
            image_path
        ):
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        classification = (
            self.classifier.predict(
                image_path=image_path,
                top_k=top_k
            )
        )

        detection = self.detector.detect(
            source=image_path,
            save=save_detection
        )

        return {
            "image": image_path,

            "classification": (
                classification
            ),

            "object_detection": (
                detection
            )
        }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "RetailMind transfer-learning "
            "classifier and YOLO detector."
        )
    )

    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the catalog image."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of product classes returned."
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="YOLO confidence threshold."
    )

    parser.add_argument(
        "--yolo-model",
        type=str,
        default="yolo11n.pt",
        help=(
            "YOLO weights. Use custom best.pt "
            "for retail-specific detection."
        )
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help=(
            "Do not save the annotated "
            "detection image."
        )
    )

    arguments = parser.parse_args()

    vision = RetailMindVision(
        detector_model=arguments.yolo_model,

        detection_threshold=(
            arguments.confidence
        )
    )

    output = vision.analyze(
        image_path=arguments.image_path,

        top_k=arguments.top_k,

        save_detection=(
            not arguments.no_save
        )
    )

    print("\n" + "=" * 70)
    print("RETAILMIND VISION RESULT")
    print("=" * 70)

    print(
        json.dumps(
            output,
            indent=4
        )
    )


if __name__ == "__main__":
    main()