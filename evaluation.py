import os
import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from model import DeepLabV3Plus
from dataset import EvalDataset

num_classes = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DeepLabV3Plus(num_classes=num_classes, weight_decay=1e-8)
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
    line = np.ones((256, 10, 3)) * 128

    input_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    ground_truth_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    ground_truth_mask = np.expand_dims(ground_truth_mask, axis=-1)
    ground_truth_mask = np.concatenate(
        [ground_truth_mask, ground_truth_mask, ground_truth_mask], axis=-1
    )
    
    prediction = np.expand_dims(prediction, axis=-1)
    prediction = np.concatenate([prediction, prediction, prediction], axis=-1)

    overlay = input_image * prediction
    prediction = prediction * 255

    final_img = np.concatenate(
        [input_image, line, ground_truth_mask, line, prediction, line, overlay], axis=1
    )

    cv2.imwrite(overlay_path, final_img)
    return


with torch.no_grad():
    for i, (images, masks) in enumerate(eval_dataloader):
        images = images.to(device)

        outputs = model(images)
        prediction = outputs.cpu().numpy()[0, 0]

        image_path = os.path.join(eval_dataset.image_dir, eval_dataset.image_filenames[i])
        mask_path = os.path.join(eval_dataset.mask_dir, eval_dataset.mask_filenames[i])

        output_image_path = os.path.join(output_dir, f"output_{i + 1}.png")

        save_overlay_image(image_path, mask_path, prediction, output_image_path)
