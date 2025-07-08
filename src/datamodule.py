import datasets
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    """Image Datasets

    Data Cards:
        - https://huggingface.co/datasets/ylecun/mnist
        - https://huggingface.co/datasets/uoft-cs/cifar10
    """

    default_transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts to tensor, scales pixel values to [0, 1]
            transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with MNIST mean & std
        ]
    )

    def __init__(
        self,
        name: str,
        split: str,
        n_classes: int = 10,
        transform: transforms.Compose | None = default_transform,
    ):
        super().__init__()

        if name == "mnist":
            self.data = datasets.load_dataset("ylecun/mnist", split=split)
            self.img_key = "image"
        elif name == "cifar10":
            self.data = datasets.load_dataset("uoft-cs/cifar10", split=split)
            self.img_key = "img"
        else:
            raise NotImplementedError("Unkwon dataset")
        # Filter to keep only samples with label < n_classes
        self.data = self.data.filter(lambda example: example["label"] < n_classes)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx][self.img_key]  # PIL Image
        label = self.data[idx]["label"]
        if self.transform is not None:
            img = self.transform(img)

        return img, label


class ImageDatamodule(L.LightningDataModule):
    def __init__(
        self,
        name: str,
        n_classes: int = 10,
        batch_size: int = 16,
        num_workers: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage):
        print(f"Preparing dataset for stage {stage}..")
        if stage in ["fit", "predict"]:
            self.dataset = ImageDataset(
                name=self.hparams.name,
                split="train",
                n_classes=self.hparams.n_classes,
            )
        else:
            self.dataset = None

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )


if __name__ == "__main__":
    dataset = ImageDataset(name="cifar10", split="train", n_classes=8, transform=None)
    print("Dataset size:", len(dataset))
    img, label = dataset[0]
    img = np.array(img)
    # np.set_printoptions(linewidth=200)  # Extend line length for print
    print("Image shape:", img.shape)
    print("Label:", label)
