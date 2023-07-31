import os
import random
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_dir, transformations=None, split="train", test_ratio=0.2):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "Image")
        self.mask_dir = os.path.join(data_dir, "Mask")
        self.image_filenames = os.listdir(self.image_dir)
        self.mask_filenames = os.listdir(self.mask_dir)
        self.transformations = transformations
        self.split = split
        self.test_ratio = test_ratio

        num_samples = len(self.image_filenames)
        num_test_samples = int(self.test_ratio * num_samples)
        
        random.seed(42)
        indices = list(range(num_samples))
        random.shuffle(indices)

        if self.split == "train":
            self.indices = indices[:-num_test_samples]
        elif self.split == "test":
            self.indices = indices[-num_test_samples:]
        else:
            raise ValueError("Invalid split value. Use 'train' or 'test'.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_idx = self.indices[idx]
        img_name = os.path.join(self.image_dir, self.image_filenames[img_idx])
        mask_name = os.path.join(self.mask_dir, self.mask_filenames[img_idx])

        image = np.array(Image.open(img_name).convert("RGB"), dtype=np.float32)
        mask = np.array(Image.open(mask_name).convert("L"), dtype=np.float32)

        image = image / 255.0
        mask = mask / 255.0

        if self.transformations is not None:
            augmentations = self.transformations(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        mask = np.expand_dims(mask, axis=0)

        return image, mask
    
class EvalDataset(Dataset):
    def __init__(self, data_dir, transformations=None):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, "Image")
        self.mask_dir = os.path.join(data_dir, "Mask")
        self.image_filenames = os.listdir(self.image_dir)
        self.mask_filenames = os.listdir(self.mask_dir)
        self.transformations = transformations

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_name = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = np.array(Image.open(img_name).convert("RGB"), dtype=np.float32)
        mask = np.array(Image.open(mask_name).convert("L"), dtype=np.float32)

        image = image / 255.0
        mask = mask / 255.0

        if self.transformations is not None:
            augmentations = self.transformations(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        mask = np.expand_dims(mask, axis=0)

        return image, mask



if __name__ == "__main__":
    data_dir = "./data_ews"
    input_size = (256, 256)

    train_transform = A.Compose(
        [
            A.Resize(input_size[0], input_size[1]),
            ToTensorV2(),
        ]
    )

    test_transform = A.Compose(
        [
            A.Resize(input_size[0], input_size[1]),
            ToTensorV2(),
        ]
    )

    # Train dataset
    train_dataset = CustomDataset(data_dir, transformations=train_transform, split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Test dataset
    test_dataset = CustomDataset(data_dir, transformations=test_transform, split="test")
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    for images, masks in train_dataloader:
        # Use the train data here for training
        print(f"Image: {images.shape}")
        print(f"Mask: {masks.shape}")
