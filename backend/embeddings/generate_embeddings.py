import torch
import open_clip
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from backend.utils.dataloader import FashionDataset
from backend.utils.dataloader import FashionDataset

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='laion2b_s34b_b79k'
)

model.eval()

dataset = FashionDataset(
    root_dir="datasets/DeepFashionSubset",
    transform=preprocess
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

all_embeddings = []
all_labels = []
all_paths = []

with torch.inference_mode():

    for batch in tqdm(loader):

        images = batch["image"]

        image_features = model.encode_image(images)

        # Normalize
        image_features /= image_features.norm(
            dim=-1,
            keepdim=True
        )

        embeddings = image_features.numpy()

        all_embeddings.append(embeddings)
        all_labels.extend(batch["label"])
        all_paths.extend(batch["path"])

all_embeddings = np.vstack(all_embeddings)
print("Embedding shape:", all_embeddings.shape)

np.save(
    "datasets/embeddings/image_embeddings.npy",
    all_embeddings
)

np.save(
    "datasets/embeddings/image_paths.npy",
    np.array(all_paths)
)

np.save(
    "datasets/embeddings/image_labels.npy",
    np.array(all_labels)
)

print("Embeddings saved.")