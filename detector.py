import os

from ultralytics import YOLO


DEFAULT_YOLO_MODEL = "yolo11n.pt"

DETECTION_DIRECTORY = "results/detections"


class ProductDetector:
    """
    Object detector based on pretrained Ultralytics YOLO.

    The default pretrained model recognizes COCO categories.
    To recognize custom retail categories, replace the model
    path with a YOLO model trained on retail annotations.
    """

    def __init__(
        self,
        model_path=DEFAULT_YOLO_MODEL,
        confidence_threshold=0.25
    ):

        self.model_path = model_path

        self.confidence_threshold = (
            confidence_threshold
        )

        # If the weights are not local, Ultralytics
        # downloads them during the first run.
        self.model = YOLO(
            model_path
        )

        os.makedirs(
            DETECTION_DIRECTORY,
            exist_ok=True
        )

    def detect(
        self,
        source,
        save=True
    ):
        """
        Detect objects and return bounding boxes.

        Parameters
        ----------
        source : str
            Image, folder, video, URL or camera source.

        save : bool
            Whether to save annotated output.

        Returns
        -------
        list
            Detection information for every input.
        """

        results = self.model.predict(
            source=source,

            conf=self.confidence_threshold,

            save=save,

            project=DETECTION_DIRECTORY,

            name="predict",

            exist_ok=True,

            verbose=False
        )

        all_results = []

        for result in results:

            image_result = {
                "source": str(
                    result.path
                ),

                "detections": []
            }

            if result.boxes is None:
                all_results.append(
                    image_result
                )
                continue

            for box in result.boxes:

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                coordinates = (
                    box.xyxy[0]
                    .cpu()
                    .tolist()
                )

                x_min = float(
                    coordinates[0]
                )

                y_min = float(
                    coordinates[1]
                )

                x_max = float(
                    coordinates[2]
                )

                y_max = float(
                    coordinates[3]
                )

                class_name = (
                    result.names[
                        class_id
                    ]
                )

                image_result[
                    "detections"
                ].append({
                    "class_id": class_id,

                    "class_name": class_name,

                    "confidence": round(
                        confidence,
                        4
                    ),

                    "bounding_box": {
                        "x_min": round(
                            x_min,
                            2
                        ),

                        "y_min": round(
                            y_min,
                            2
                        ),

                        "x_max": round(
                            x_max,
                            2
                        ),

                        "y_max": round(
                            y_max,
                            2
                        ),

                        "width": round(
                            x_max - x_min,
                            2
                        ),

                        "height": round(
                            y_max - y_min,
                            2
                        )
                    }
                })

            all_results.append(
                image_result
            )

        return all_results