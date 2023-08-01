import csv
import albumentations as A
import torch
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import DeepLabV3Plus
from metrics import DiceLoss, calculate_metrics

data_dir = "./data_ews"
input_size = (256, 256)
batch_size = 4
num_classes = 1
learning_rate = 1e-4
num_epochs = 80

train_transform = A.Compose(
    [
        A.Resize(input_size[0], input_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.Resize(input_size[0], input_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

train_dataset = CustomDataset(data_dir, transformations=train_transform, split="train")
test_dataset = CustomDataset(data_dir, transformations=test_transform, split="test")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = DeepLabV3Plus(num_classes=num_classes)

criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.1, verbose=True
)


csv_file = "training_logs.csv"
csv_header = [
    "Epoch",
    "Avg Train Loss",
    "Avg Val Loss",
    "Avg IoU Train",
    "Avg IoU Val",
    "Avg Pix Acc Train",
    "Avg Pix Acc Val",
    "Avg Dice Coeff Train",
    "Avg Dice Coeff Val",
]

best_val_loss = float("inf")

with open(csv_file, "w", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(csv_header)

    for epoch in range(num_epochs):
        # TRAINING
        model.train()
        train_loss = 0.0
        total_iou_train = 0.0
        total_pixel_accuracy_train = 0.0
        total_dice_coefficient_train = 0.0

        train_dataloader = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"
        )

        for images, masks in train_dataloader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            t_loss = criterion(torch.sigmoid(outputs), masks)

            t_loss.backward()
            optimizer.step()

            train_loss += t_loss.item()

            # Calculating metrics for training
            with torch.no_grad():
                pred_masks = torch.sigmoid(outputs) > 0.5
                iou_train, dice_coefficient_train, pixel_accuracy_train = calculate_metrics(
                    pred_masks, masks
                )

                total_iou_train += iou_train
                total_dice_coefficient_train += dice_coefficient_train
                total_pixel_accuracy_train += pixel_accuracy_train

            # Displaying metrics in the progress bar description
            train_dataloader.set_postfix(
                loss=t_loss.item(),
                train_iou=iou_train,
                train_pix_acc=pixel_accuracy_train,
                train_dice_coef=dice_coefficient_train,
            )

        train_loss /= len(train_dataloader)
        avg_iou_train = total_iou_train / len(train_dataloader)
        avg_pixel_accuracy_train = total_pixel_accuracy_train / len(train_dataloader)
        avg_dice_coefficient_train = total_dice_coefficient_train / len(train_dataloader)

        # VALIDATION
        model.eval()
        val_loss = 0.0
        total_iou_val = 0.0
        total_pixel_accuracy_val = 0.0
        total_dice_coefficient_val = 0.0

        test_dataloader = tqdm(test_dataloader, desc=f"Validation", unit="batch")

        with torch.no_grad():
            for images, masks in test_dataloader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)

                v_loss = criterion(torch.sigmoid(outputs), masks)
                val_loss += v_loss.item()

                # Calculating metrics for Validation
                pred_masks = torch.sigmoid(outputs) > 0.5
                iou_val, dice_coefficient_val, pixel_accuracy_val = calculate_metrics(
                    pred_masks, masks
                )

                total_iou_val += iou_val
                total_pixel_accuracy_val += pixel_accuracy_val
                total_dice_coefficient_val += dice_coefficient_val

                # Displaying metrics in progress bar description
                test_dataloader.set_postfix(
                    val_loss=v_loss.item(),
                    val_iou=iou_val,
                    val_pix_acc=pixel_accuracy_val,
                    val_dice_coef=dice_coefficient_val,
                )

        val_loss /= len(test_dataloader)
        avg_iou_val = total_iou_val / len(test_dataloader)
        avg_pixel_accuracy_val = total_pixel_accuracy_val / len(test_dataloader)
        avg_dice_coefficient_val = total_dice_coefficient_val / len(test_dataloader)

        scheduler.step(val_loss)

        print(
            f"\nEpoch {epoch + 1}/{num_epochs}\n"
            f"Avg Train Loss: {train_loss:.4f}\n"
            f"Avg Validation Loss: {val_loss:.4f}\n"
            f"Avg IoU Train: {avg_iou_train:.4f}\n"
            f"Avg IoU Val: {avg_iou_val:.4f}\n"
            f"Avg Pix Acc Train: {avg_dice_coefficient_train:.4f}\n"
            f"Avg Pix Acc Val: {avg_pixel_accuracy_val:.4f}\n"
            f"Avg Dice Coeff Train: {avg_dice_coefficient_train:.4f}\n"
            f"Avg Dice Coeff Val: {avg_dice_coefficient_val:.4f}\n"
            f"{'-'*50}"
        )

        if val_loss < best_val_loss:
            print(
                f"\nCurrent Validation Loss: {val_loss:.4f} is better than previous Validation Loss: {best_val_loss:.4f}\n"
            )
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved Best Model!\n")
            print(f"{'-'*50}")

        # Append the training and validation logs to the CSV file
        csv_writer.writerow(
            [
                epoch + 1,
                train_loss,
                val_loss,
                avg_iou_train,
                avg_iou_val,
                avg_pixel_accuracy_train,
                avg_pixel_accuracy_val,
                avg_dice_coefficient_train,
                avg_dice_coefficient_val,
            ]
        )
