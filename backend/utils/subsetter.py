from pathlib import Path
import shutil
import random

SRC_DIR = Path("datasets/DeepFashion")
TGT_DIR = Path("datasets/DeepFashionSubset")

MAX_CLASSES = 200
IMAGES_PER_CLASS = 10

def create_subset():
    TGT_DIR.mkdir(parents=True, exist_ok=True)
    class_folders = [
        folder for folder in SRC_DIR.iterdir()
        if folder.is_dir()
    ]
    class_folders = sorted(class_folders)
    selected_classes = random.sample(
        class_folders,
        min(MAX_CLASSES, len(class_folders))
    )
    total_images = 0
    for class_folder in selected_classes:

        target_class_dir = TGT_DIR / class_folder.name
        target_class_dir.mkdir(parents=True, exist_ok=True)

        images = list(class_folder.glob("*.jpg"))

        selected_images = random.sample(
            images,
            min(IMAGES_PER_CLASS, len(images))
        )
        for img_path in selected_images:

            shutil.copy2(
                img_path,
                target_class_dir / img_path.name
            )

            total_images += 1

    print(f"Created subset with {total_images} images")


if __name__ == "__main__":
    create_subset()