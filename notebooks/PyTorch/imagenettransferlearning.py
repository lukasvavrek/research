import torch
from pl_bolts.models.self_supervised import SimCLR
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import functional as F
from torch.optim import Adam
import torchmetrics


class ImagenetTransferLearning(LightningModule):
    # network input
    # Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def __init__(self, extract_features=False):
        super().__init__()

        self.extract_features = extract_features

        # init a pretrained resnet
        weight_path = r'D:\Users\lVavrek\research\data\sim-clr-backups\06122021-self-supervised-librispeech-epoch=34-val_loss=1.21.ckpt'
        simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
        backbone = simclr.encoder

        # extract last layer
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*layers)

        # use the pretrained model to classify destination problem (PD or healthy, 2 classes)
        num_target_classes = 2
        self.classifier = torch.nn.Linear(num_filters, num_target_classes)
        self.sigmoid = torch.nn.Sigmoid()

        self.celoss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        # self.feature_extractor.eval() # this freezes the layers
        with torch.no_grad():
            x = x[0]  # hack, because we use SimCLR transform which returns 3 augmented images; use custom tranform instead
            representations = self.feature_extractor(x).flatten(1)

        if self.extract_features:
            return representations

        x = self.classifier(representations)
        x = self.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.celoss(logits, y)  # F.nll_loss(logits, y)
        self.log("loss", loss)
        return loss

    # https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = torchmetrics.accuracy(y_hat, y)
        return loss, acc

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.0001)  # , lr=0.001
