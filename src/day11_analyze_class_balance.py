import csv
import json
import os


TRAIN_DIRECTORY = "data/fashion_split/train"

DISTRIBUTION_PATH = (
    "results/class_distribution.csv"
)

CONFIG_PATH = (
    "results/rare_class_config.json"
)

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


if not os.path.isdir(TRAIN_DIRECTORY):
    raise FileNotFoundError(
        f"Training directory not found: "
        f"{TRAIN_DIRECTORY}\n"
        "Run prepare_vision_split.py first."
    )


os.makedirs(
    "results",
    exist_ok=True
)


class_counts = {}


for class_name in sorted(
    os.listdir(TRAIN_DIRECTORY)
):

    class_directory = os.path.join(
        TRAIN_DIRECTORY,
        class_name
    )

    if not os.path.isdir(class_directory):
        continue

    image_count = sum(
        1
        for file_name in os.listdir(
            class_directory
        )
        if file_name.lower().endswith(
            IMAGE_EXTENSIONS
        )
    )

    if image_count > 0:
        class_counts[class_name] = image_count


if not class_counts:
    raise ValueError(
        "No product images were found."
    )


rare_class = min(
    class_counts,
    key=class_counts.get
)

rare_class_count = class_counts[
    rare_class
]

largest_class_count = max(
    class_counts.values()
)

# Generate enough images to match the largest category.
required_synthetic_count = max(
    0,
    largest_class_count
    - rare_class_count
)


with open(
    DISTRIBUTION_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "class_name",
        "image_count",
        "is_rare_class"
    ])

    for class_name, image_count in sorted(
        class_counts.items(),
        key=lambda item: item[1]
    ):

        writer.writerow([
            class_name,
            image_count,
            class_name == rare_class
        ])


configuration = {
    "rare_class": rare_class,
    "rare_class_count": rare_class_count,
    "largest_class_count": largest_class_count,
    "required_synthetic_count": (
        required_synthetic_count
    ),
    "training_directory": TRAIN_DIRECTORY
}


with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        configuration,
        file,
        indent=4
    )


print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)


for class_name, image_count in sorted(
    class_counts.items(),
    key=lambda item: item[1]
):

    marker = (
        " <- RARE CLASS"
        if class_name == rare_class
        else ""
    )

    print(
        f"{class_name:<25}: "
        f"{image_count}{marker}"
    )


print("\nRare class:", rare_class)

print(
    "Rare-class image count:",
    rare_class_count
)

print(
    "Largest-class image count:",
    largest_class_count
)

print(
    "Synthetic images required:",
    required_synthetic_count
)

print(
    "\nDistribution saved:",
    DISTRIBUTION_PATH
)

print(
    "Configuration saved:",
    CONFIG_PATH
)