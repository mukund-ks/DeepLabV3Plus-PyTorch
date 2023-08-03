import os
import torch
import albumentations as A
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from model import DeepLabV3Plus
from dataset import EvalDataset
from utils import save_overlay_image

# TODO:
#   * Refactor Code
#   * Write cli

num_classes = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DeepLabV3Plus(num_classes=num_classes)
model.load_state_dict(torch.load("best_model.pth"))
model.to(device)
model.eval()

output_dir = "./eval_output"
os.makedirs(output_dir, exist_ok=True)

eval_data_dir = "./eval_ews"

eval_transform = A.Compose(
    [
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)
eval_dataset = EvalDataset(data_dir=eval_data_dir, transformations=eval_transform)
eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

with torch.no_grad():
    eval_dataloader = tqdm(eval_dataloader, desc="Evaluation", unit="image")
    for i, (images, masks) in enumerate(eval_dataloader):
        images = images.to(device)

        outputs = model(images)
        prediction = outputs.cpu().numpy()[0, 0]

        image_path = eval_dataset.image_filenames[i]
        mask_path = eval_dataset.mask_filenames[i]

        output_image_path = os.path.join(output_dir, f"output_{i + 1}.png")

        save_overlay_image(image_path, mask_path, prediction, output_image_path)
