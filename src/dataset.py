import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MNISTDataset(Dataset):
    """MNISTDataset

    Data Card: https://huggingface.co/datasets/ylecun/mnist
    """

    default_transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Converts to tensor, scales pixel values to [0, 1]
            transforms.Normalize((0.1307,), (0.3081,)),  # Normalize with MNIST mean & std
        ]
    )

    def __init__(self, split: str, transform: transforms.Compose | None = default_transform):
        super().__init__()

        self.data = datasets.load_dataset("ylecun/mnist", split=split)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]["image"]  # PIL Image
        label = self.data[idx]["label"]
        if self.transform is not None:
            img = self.transform(img)

        return img, label


if __name__ == "__main__":
    dataset = MNISTDataset(split="train", transform=None)
    img, label = dataset[0]
    img = np.array(img)
    np.set_printoptions(linewidth=200)  # Extend line length for print
    print(img)
    print("Label:", label)
