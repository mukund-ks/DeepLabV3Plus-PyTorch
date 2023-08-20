import os
import sys
import torch
import click
import traceback
import albumentations as A
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from model import DeepLabV3Plus
from dataset import EvalDataset
from utils import save_overlay_image

CLASSES = 1
INPUT = (256, 256)


@click.command()
@click.option("-D", "--data-dir", type=str, required=True, help="Path for Data Directory")
def main(data_dir: str) -> None:
    """
    Evaluation Script for DeepLabV3+ with ResNet50 Encoder for Binary Segmentation.\n
    Please make sure your evaluation data is structured according to the folder structure specified in the Github Repository.\n
    See: https://github.com/mukund-ks/DeepLabV3Plus-PyTorch

    Refer to the Option(s) below for usage.
    """
    click.secho(message="🔎 Evaluation...", fg="blue")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepLabV3Plus(num_classes=CLASSES)
    model.load_state_dict(torch.load("./output/best_model.pth"))
    model.to(device)
    model.eval()

    output_dir = "./eval_output"
    os.makedirs(output_dir, exist_ok=True)

    eval_transform = A.Compose(
        [
            A.Resize(INPUT[0], INPUT[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    try:
        eval_dataset = EvalDataset(data_dir=data_dir, transformations=eval_transform)
        eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
        click.echo(message=f"\n{click.style('Evaluation Size: ', fg='blue')}{eval_dataset.__len__()}\n")
        eval_dataloader = tqdm(eval_dataloader, desc="Evaluation", unit="image")
    except Exception as _:
        click.secho(message="\n❗ Error\n", fg="red")
        click.secho(message=traceback.format_exc(), fg="yellow")
        sys.exit("Non-Existent Data Dir")

    with torch.no_grad():
        for i, (image, _) in enumerate(eval_dataloader):
            image = image.to(device)

            output = model(image)
            prediction = output.cpu().numpy()[0, 0]

            img_path = eval_dataset.image_filenames[i]
            mask_path = eval_dataset.mask_filenames[i]

            output_img_path = os.path.join(output_dir, f"output_{i + 1}.png")

            save_overlay_image(img_path, mask_path, prediction, output_img_path)

    click.secho(message="🎉 Evaluation Done!", fg="blue")
    return


if __name__ == "__main__":
    main()
