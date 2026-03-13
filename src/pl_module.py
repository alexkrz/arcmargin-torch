import lightning as L
import torch

from src.models.backbones import ConvNet, LeNetExt
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
        data_name: str = "mnist",
        header: str = "linear",
        embed_dim: int = 3,
        n_classes: int = 10,
        s: float | None = None,
        m: float | None = None,
        lr: float = 0.01,
    ):
        super().__init__()
        if data_name == "mnist":
            # self.backbone = LeNetExt(in_channels=1, input_size=28, n_features=embed_dim)
            self.backbone = ConvNet(in_channels=1, n_features=embed_dim)
        elif data_name == "cifar10":
            # self.backbone = LeNetExt(in_channels=3, input_size=32, n_features=embed_dim)
            self.backbone = ConvNet(in_channels=3, n_features=embed_dim)
        else:
            raise NotImplementedError("Unknown data_name")
        assert header in header_dict.keys()
        kwargs = {k: v for k, v in (("s", s), ("m", m)) if v is not None}
        self.header = header_dict[header](embed_dim, n_classes, **kwargs)
        s = self.header.s
        m = self.header.m

        self.save_hyperparameters()

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
        self.log("loss", loss, prog_bar=True, on_epoch=True)
        self.log("optimizer_lr", optimizer_lr)
        self.log("max_ampl", max_ampl.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            # Need to optimize over all parameters in the module!
            params=self.parameters(),
            lr=self.hparams.lr,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.25)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
