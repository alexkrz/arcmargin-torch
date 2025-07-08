from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from jsonargparse import CLI
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from src.datamodule import ImageDatamodule
from src.pl_module import ArcMarginModule, header_dict


def plot_3d(embeds, labels, fig_path="./example.png"):
    fig = plt.figure(figsize=(10, 10))
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0 : 2.0 * pi : 100j]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color="w", alpha=0.3, linewidth=0)
    ax.scatter(embeds[:, 0], embeds[:, 1], embeds[:, 2], c=labels, s=20)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(fig_path)


def plot_2d(embeds, labels, fig_path="./example.png"):
    """
    Plots 2D embeddings on the unit circle, colored by label.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit circle
    circle = plt.Circle((0, 0), 1, color="gray", fill=False, linestyle="--", linewidth=2)
    ax.add_artist(circle)

    scatter = ax.scatter(embeds[:, 0], embeds[:, 1], c=labels, s=20, cmap="tab10", alpha=0.8)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)


def main(
    chkpt_dir: str = "lightning_logs",
    header: str = "linear",
    version: int | None = None,
):
    """Compare embeddings from different Headers on Hypersphere

    Args:
        chkpt_dir (str, optional): The checkpoint root directory.
        header (str, optional): The header name.
    """

    assert header in header_dict.keys(), "Unknown header"
    chkpt_dir = Path(chkpt_dir) / header  # type: Path
    if version is not None:
        chkpt_dir = chkpt_dir / f"version_{version}"
    chkpt_files = sorted(list(chkpt_dir.rglob("*.ckpt")))
    chkpt_fp = chkpt_files[-1]
    print("Chkpt:", str(chkpt_fp))

    # Assign device where code is executed
    if torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Neural Engine (MPS)
    else:
        device = torch.device("cpu")  # Default to CPU
    print("Device:", device)

    # chkpt = torch.load(chkpt_fp, weights_only=True)
    # print(chkpt.keys())
    pl_module = ArcMarginModule.load_from_checkpoint(chkpt_fp)
    backbone = pl_module.backbone

    datamodule = ImageDatamodule.load_from_checkpoint(chkpt_fp)
    datamodule.setup("predict")
    dataloader = datamodule.predict_dataloader()

    backbone.to(device)
    backbone.eval()

    feats_list = []
    labels_list = []
    for batch in tqdm(dataloader):
        imgs, labels = batch
        with torch.no_grad():
            feats = backbone(imgs.to(device))
        feats = F.normalize(feats)  # Normalize feats to unit length
        feats_list.append(feats.cpu().numpy())  # Move feats to CPU and convert to numpy array
        labels_list.append(labels.cpu().numpy())

    feats = np.concatenate(feats_list)
    labels = np.concatenate(labels_list)

    print(feats.shape)
    print(labels.shape)

    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    # plot_3d(feats, labels, fig_path=results_dir / f"{header}.png")
    plot_2d(feats, labels, fig_path=results_dir / f"{header}.png")


if __name__ == "__main__":
    CLI(main)
