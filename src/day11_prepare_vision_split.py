import os
import random
import shutil


SOURCE_DIRECTORY = "data/fashion_products"

SPLIT_DIRECTORY = "data/fashion_split"

TRAIN_DIRECTORY = os.path.join(
    SPLIT_DIRECTORY,
    "train"
)

VALIDATION_DIRECTORY = os.path.join(
    SPLIT_DIRECTORY,
    "validation"
)

VALIDATION_RATIO = 0.20

RANDOM_STATE = 42

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


random.seed(RANDOM_STATE)


if not os.path.isdir(SOURCE_DIRECTORY):
    raise FileNotFoundError(
        f"Dataset not found: {SOURCE_DIRECTORY}"
    )


# Remove the previous split so that rerunning the script
# creates a clean, reproducible split.
if os.path.exists(SPLIT_DIRECTORY):
    shutil.rmtree(SPLIT_DIRECTORY)


os.makedirs(
    TRAIN_DIRECTORY,
    exist_ok=True
)

os.makedirs(
    VALIDATION_DIRECTORY,
    exist_ok=True
)


class_names = sorted([
    folder_name
    for folder_name in os.listdir(
        SOURCE_DIRECTORY
    )
    if os.path.isdir(
        os.path.join(
            SOURCE_DIRECTORY,
            folder_name
        )
    )
])


if len(class_names) < 2:
    raise ValueError(
        "At least two category folders are required."
    )


print("=" * 70)
print("CREATING FIXED VISION DATA SPLIT")
print("=" * 70)


for class_name in class_names:

    source_class_directory = os.path.join(
        SOURCE_DIRECTORY,
        class_name
    )

    train_class_directory = os.path.join(
        TRAIN_DIRECTORY,
        class_name
    )

    validation_class_directory = os.path.join(
        VALIDATION_DIRECTORY,
        class_name
    )

    os.makedirs(
        train_class_directory,
        exist_ok=True
    )

    os.makedirs(
        validation_class_directory,
        exist_ok=True
    )

    image_files = sorted([
        file_name
        for file_name in os.listdir(
            source_class_directory
        )
        if file_name.lower().endswith(
            IMAGE_EXTENSIONS
        )
    ])

    if len(image_files) < 2:
        print(
            f"Skipping {class_name}: "
            "not enough images."
        )
        continue

    random.shuffle(image_files)

    validation_count = max(
        1,
        int(
            len(image_files)
            * VALIDATION_RATIO
        )
    )

    validation_files = image_files[
        :validation_count
    ]

    training_files = image_files[
        validation_count:
    ]

    for file_name in training_files:

        shutil.copy2(
            os.path.join(
                source_class_directory,
                file_name
            ),
            os.path.join(
                train_class_directory,
                file_name
            )
        )

    for file_name in validation_files:

        shutil.copy2(
            os.path.join(
                source_class_directory,
                file_name
            ),
            os.path.join(
                validation_class_directory,
                file_name
            )
        )

    print(
        f"{class_name:<25} "
        f"train={len(training_files):<6} "
        f"validation={len(validation_files)}"
    )


print("\nSplit created:")

print(TRAIN_DIRECTORY)
print(VALIDATION_DIRECTORY)