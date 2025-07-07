import lightning as L
import torch

from src.models.backbones import ConvNet
from src.models.headers import ArcFaceHeader, CosFaceHeader, LinearHeader, SphereFaceHeader

header_dict = {
    "linear": LinearHeader,
    "sphereface": SphereFaceHeader,
    "cosface": CosFaceHeader,
    "arcface": ArcFaceHeader,
}


class ArcMarginModule(L.LightningModule):
    def __init__(
        self,
        header: str = "linear",
        embed_dim: int = 3,
        n_classes: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = ConvNet(embed_dim)
        assert header in header_dict.keys()
        self.header = header_dict[header](embed_dim, n_classes)

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(imgs)
        return feats

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        feats = self(imgs)
        ampl = torch.norm(feats, dim=1)
        max_ampl = torch.max(ampl)
        logits = self.header(feats, targets)
        # logits vector describes the probability for each image to belong to one of n_classes
        loss = self.criterion(logits, targets)
        optimizer_lr = self.optimizers().optimizer.param_groups[0]["lr"]
        self.log("loss", loss, prog_bar=True)
        self.log("optimizer_lr", optimizer_lr)
        self.log("max_ampl", max_ampl.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            # Need to optimize over all parameters in the module!
            params=self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
