import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from src.gradcam import (
    model,
    backbone,
    class_names,
    LAST_CONV_LAYER,
    load_image,
    make_gradcam_heatmap,
    create_overlay
)


OUTPUT_DIRECTORY = "results/gradcam"

os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)


def explain_product(
    image_path,
    alpha=0.40
):

    # --------------------------------------------------------
    # 1. LOAD IMAGE
    # --------------------------------------------------------

    original_image, image_batch = (
        load_image(image_path)
    )


    # --------------------------------------------------------
    # 2. GRAD-CAM + PREDICTION
    # --------------------------------------------------------

    (
        heatmap,
        probabilities,
        class_index
    ) = make_gradcam_heatmap(
        image_batch=image_batch,

        model=model,

        backbone=backbone,

        last_conv_layer_name=(
            LAST_CONV_LAYER
        )
    )


    # --------------------------------------------------------
    # 3. PREDICTION INFORMATION
    # --------------------------------------------------------

    predicted_class = (
        class_names[
            class_index
        ]
    )

    confidence = float(
        probabilities[
            class_index
        ]
    )


    # --------------------------------------------------------
    # 4. CREATE HEATMAP OVERLAY
    # --------------------------------------------------------

    overlay = create_overlay(
        original_image,
        heatmap,
        alpha=alpha
    )


    # --------------------------------------------------------
    # 5. SAVE EXPLANATION
    # --------------------------------------------------------

    file_name = os.path.basename(
        image_path
    )

    file_stem = os.path.splitext(
        file_name
    )[0]

    output_path = os.path.join(
        OUTPUT_DIRECTORY,
        f"{file_stem}_gradcam.png"
    )


    # --------------------------------------------------------
    # 6. DISPLAY 3-PANEL EXPLANATION
    # --------------------------------------------------------

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    # Original
    axes[0].imshow(
        original_image
    )

    axes[0].set_title(
        "Original Image"
    )

    axes[0].axis("off")


    # Heatmap
    axes[1].imshow(
        heatmap,
        cmap="jet"
    )

    axes[1].set_title(
        "Grad-CAM Heatmap"
    )

    axes[1].axis("off")


    # Overlay
    axes[2].imshow(
        overlay.astype("uint8")
    )

    axes[2].set_title(
        f"{predicted_class}\n"
        f"Confidence: {confidence:.4f}"
    )

    axes[2].axis("off")


    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )


    # --------------------------------------------------------
    # 7. RESULT
    # --------------------------------------------------------

    result = {
        "predicted_category": (
            predicted_class
        ),

        "confidence": round(
            confidence,
            4
        ),

        "gradcam_image": (
            output_path
        )
    }

    return result


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Explain RetailMind product "
            "classification using Grad-CAM."
        )
    )

    parser.add_argument(
        "image_path",
        type=str
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.40
    )

    args = parser.parse_args()

    result = explain_product(
        args.image_path,
        alpha=args.alpha
    )

    print(
        "\nPrediction:"
    )

    print(
        "Category:",
        result["predicted_category"]
    )

    print(
        "Confidence:",
        result["confidence"]
    )

    print(
        "Grad-CAM:",
        result["gradcam_image"]
    )
