import os
import csv
import sys
import click
import traceback
import albumentations as A
import torch
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import DeepLabV3Plus
from metrics import DiceLoss, calculate_metrics

INPUT = (256, 256)
CLASSES = 1  # Binary Segmentation


@click.command()
@click.option("-D", "--data-dir", type=str, required=True, help="Path for Data Directory")
@click.option(
    "-E",
    "--num-epochs",
    type=int,
    default=25,
    help="Number of epochs to train the model for. Default - 25",
)
@click.option(
    "-L",
    "--learning-rate",
    type=float,
    default=1e-4,
    help="Learning Rate for model. Default - 1e-4",
)
@click.option(
    "-B",
    "--batch-size",
    type=int,
    default=4,
    help="Batch size of data for training. Default - 4",
)
@click.option(
    "-P",
    "--pre-split",
    required=True,
    type=bool,
    help="Opt-in to split data into Training and Validaton set.",
)
@click.option(
    "-A",
    "--augment",
    type=bool,
    default=True,
    help="Opt-in to apply augmentations to training set. Default - True",
)
@click.option(
    "-S",
    "--early-stop",
    type=bool,
    default=True,
    help="Stop training if val_loss hasn't improved for a certain no. of epochs. Default - True",
)
def main(
    data_dir: str,
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    pre_split: bool,
    augment: bool,
    early_stop: bool,
) -> None:
    """
    Training Script for DeepLabV3+ with ResNet50 Encoder for Binary Segmentation.\n
    Please make sure your data is structured according to the folder structure specified in the Github Repository.\n
    See: https://github.com/mukund-ks/DeepLabV3Plus-PyTorch

    Refer to the Options below for usage.
    """
    click.secho(message="🚀 Training...", fg="blue", nl=True)

    os.makedirs("output", exist_ok=True)

    train_transform_list = [
        A.Resize(INPUT[0], INPUT[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]

    test_transform_list = [
        A.Resize(INPUT[0], INPUT[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]

    if augment:
        train_transform_list = (
            train_transform_list[:1]
            + [
                A.Rotate(limit=(-10, 10), p=0.7),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomCrop(height=INPUT[0], width=INPUT[0], p=0.7),
            ]
            + train_transform_list[1:]
        )
    else:
        pass

    train_transform = A.Compose(transforms=train_transform_list, is_check_shapes=False)

    test_transform = A.Compose(transforms=test_transform_list, is_check_shapes=False)

    try:
        if pre_split:
            train_dataset = CustomDataset(
                data_dir=os.path.join(data_dir, "Train"),
                transformations=train_transform,
                pre_split=True,
            )
            test_dataset = CustomDataset(
                data_dir=os.path.join(data_dir, "Test"),
                transformations=test_transform,
                pre_split=True,
            )
        else:
            train_dataset = CustomDataset(
                data_dir=data_dir, transformations=train_transform, split="train"
            )
            test_dataset = CustomDataset(
                data_dir=data_dir, transformations=test_transform, split="test"
            )
    except Exception as _:
        click.secho(message="\n❗ Error \n", fg="red")
        click.secho(message=traceback.format_exc(), fg="yellow")
        sys.exit("Non-Existent Data Dir")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Defining Model
    model = DeepLabV3Plus(num_classes=CLASSES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = DiceLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.1, verbose=True
    )

    # For Early-Stopping
    patience_epochs = 20
    no_improvement_epochs = 0

    # Logging
    csv_file = os.path.abspath("output/training_logs.csv")
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
        "Learning Rate",
    ]

    # For saving best model
    best_val_loss = float("inf")

    click.echo(
        f"\n{click.style(text=f'Train Size: ', fg='blue')}{train_dataset.__len__()}\t{click.style(text=f'Test Size: ', fg='blue')}{test_dataset.__len__()}\n"
    )

    # Main loop
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

            current_lr = optimizer.param_groups[0]["lr"]

            for images, masks in train_dataloader:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                t_loss = criterion(outputs, masks)

                t_loss.backward()
                optimizer.step()

                train_loss += t_loss.item()

                # Calculating metrics for training
                with torch.no_grad():
                    pred_masks = outputs > 0.5
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
                    lr=current_lr,
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

                    v_loss = criterion(outputs, masks)
                    val_loss += v_loss.item()

                    # Calculating metrics for Validation
                    pred_masks = outputs > 0.5
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
                        lr=current_lr,
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
                f"Current LR: {current_lr}\n"
                f"{'-'*50}"
            )

            # Saving best model
            if val_loss < best_val_loss:
                no_improvement_epochs = 0
                click.secho(
                    message=f"\n👀 val_loss improved from {best_val_loss:.4f} to {val_loss:.4f}\n",
                    fg="green",
                )
                best_val_loss = val_loss
                torch.save(model.state_dict(), "./output/best_model.pth")
                click.secho(message="Saved Best Model! 🙌\n", fg="green")
                print(f"{'-'*50}")
            else:
                no_improvement_epochs += 1
                click.secho(
                    message=f"\nval_loss did not improve from {best_val_loss:.4f}\n", fg="yellow"
                )
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
                    current_lr,
                ]
            )

            # Early-Stopping
            if early_stop:
                if no_improvement_epochs >= patience_epochs:
                    click.secho(
                        message=f"\nEarly Stopping: val_loss did not improve for {patience_epochs} epochs.\n",
                        fg="red",
                    )
                    break

    click.secho(message="🎉 Training Done!", fg="blue", nl=True)

    return


if __name__ == "__main__":
    main()
