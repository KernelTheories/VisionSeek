from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import random

class FashionDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.samples = []
        class_folders = [
            folder for folder in self.root_dir.iterdir()
            if folder.is_dir()
        ]

        for class_folder in class_folders:

            label = class_folder.name.replace("_", " ").lower()

            images = list(class_folder.glob("*.jpg"))

            for img_path in images:

                self.samples.append({
                    "image_path": img_path,
                    "label": label
                })

        print(f"Loaded {len(self.samples)} images")


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):

        sample = self.samples[idx]

        image = Image.open(
            sample["image_path"]
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": sample["label"],
            "path": str(sample["image_path"])
        }