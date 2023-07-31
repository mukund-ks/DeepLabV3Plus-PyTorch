import os
import numpy as np
import torch
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from model import DeepLabV3Plus
from dataset import EvalDataset
from PIL import Image

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


def save_overlay_image(image_path, mask_path, prediction, overlay_path):
    input_image = Image.open(image_path).convert("RGB")
    ground_truth_mask = Image.open(mask_path).convert("L")
    predicted_mask = Image.fromarray((prediction * 255).astype(np.uint8))

    # Overlay the prediction on the input image
    overlay = Image.blend(input_image, predicted_mask.convert("RGB"), alpha=0.5)

    # Arrange the images horizontally and save the result
    result_image = Image.new("RGB", (input_image.width * 4, input_image.height))
    result_image.paste(input_image, (0, 0))
    result_image.paste(ground_truth_mask, (input_image.width, 0))
    result_image.paste(predicted_mask.convert("RGB"), (input_image.width*2, 0))
    result_image.paste(overlay, (input_image.width * 3, 0))

    result_image.save(overlay_path)


with torch.no_grad():
    for i, (images, masks) in enumerate(eval_dataloader):
        images = images.to(device)
        
        outputs = model(images)
        prediction = torch.sigmoid(outputs).cpu().numpy()[0, 0]
        
        image_path = os.path.join(eval_dataset.image_dir, eval_dataset.image_filenames[i])
        mask_path = os.path.join(eval_dataset.mask_dir, eval_dataset.mask_filenames[i])

        output_image_path = os.path.join(output_dir, f"output_{i + 1}.png")

        save_overlay_image(image_path, mask_path, prediction, output_image_path)
