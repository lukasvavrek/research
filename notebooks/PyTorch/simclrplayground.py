# https://pytorch.org/audio/stable/_modules/torchaudio/datasets/librispeech.html#LIBRISPEECH

# this script is pretty much a copy of SIM CLR playground.ipynb
# I created it to try avoid those bugs:
# https://github.com/PyTorchLightning/lightning-bolts/issues/640
# https://github.com/PyTorchLightning/lightning-bolts/issues/642

# https://github.com/PyTorchLightning/Lightning-Bolts/blob/master/pl_bolts/models/self_supervised/simclr/transforms.py
# https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.loggers.TensorBoardLogger.html#pytorch_lightning.loggers.TensorBoardLogger

import os

from numpy.lib.scimath import log
import torch
from torch.nn import functional as F
from pytorch_lightning.metrics import functional as FM
from torch.optim import Adam
from torchaudio.transforms import Spectrogram, AmplitudeToDB # try using mel spectrogram as well
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform
from pytorch_lightning.trainer import Trainer
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule

from pl_bolts.datamodules import CIFAR10DataModule

from PIL import Image
import matplotlib.pyplot as plt

import torchvision
from torchvision import transforms
import torchaudio
import numpy as np
import sys

from skimage.util import img_as_ubyte
from skimage import exposure
from sklearn import preprocessing

from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter

# fixing imports TODO: fix this 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pcgita.dataset import PcGitaTorchDataset
from librispeech.dataset import LibrispeechSpectrogramDataset

def train_self_supervised():
    logger = TensorBoardLogger('runs', name='SimCLR_libri_speech')

    # 8, 224, 8 worked well
    # 16, 224, 4 as well
    batch_size = 16
    input_height = 224
    num_workers = 4

    train_dataset = LibrispeechSpectrogramDataset(transform=SimCLRTrainDataTransform(input_height=input_height, gaussian_blur=False), train=True)
    val_dataset = LibrispeechSpectrogramDataset(transform=SimCLREvalDataTransform(input_height=input_height, gaussian_blur=False), train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    model = SimCLR(gpus=1, num_samples=len(train_dataset), batch_size=batch_size, dataset='librispeech')

    ##
    
    td = LibrispeechSpectrogramDataset(transform=None, train=True)
    samples = []
    toTensor = transforms.ToTensor()
    for i in range(0, 24):
        img, cls = td.__getitem__(i)
        img = toTensor(img)
        samples.append(img)

    grid = torchvision.utils.make_grid(samples, padding=10, nrow=4)
    logger.experiment.add_image("generated_images", grid, 0)
    logger.finalize("success")

    return

    ##

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=r'D:\Users\lVavrek\research\data',
        filename="self-supervised-librispeech-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    early_stopping = EarlyStopping(monitor="val_loss")

    trainer = Trainer(gpus=1, callbacks=[checkpoint_callback, early_stopping], logger=logger)
    trainer.fit(model, train_loader, test_loader)

def train_transfer_learning():
    logger = TensorBoardLogger('runs', name='pc-gita')

    batch_size = 32
    input_height = 224
    num_workers = 4

    train_dataset = PcGitaTorchDataset(transform=SimCLRTrainDataTransform(input_height=input_height, gaussian_blur=False), train=True)
    val_dataset = PcGitaTorchDataset(transform=SimCLRTrainDataTransform(input_height=input_height, gaussian_blur=False), train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    model = ImagenetTransferLearning()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=r'D:\Users\lVavrek\research\data',
        filename="transfer-learning-pcgita-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    # early_stopping = EarlyStopping(monitor="val_loss")

    trainer = Trainer(gpus=1, callbacks=[checkpoint_callback], logger=logger, max_epochs=20)
    trainer.fit(model, train_loader, test_loader)

    # val_dataset = LibrispeechSpectrogramDataset(transform=SimCLREvalDataTransform(input_height=224, gaussian_blur=False), train=False)
    # test_loader = DataLoader(val_dataset, batch_size=16, num_workers=4)

    # net = ImagenetTransferLearning()
    # x = next(iter(test_loader))[0][0][:1] # single sample, drop the label; TODO: check what is returned by data loader
    # print(len(x))
    # out = net(x)
    # print(out)
    # print(net.feature_extractor.eval())

class ImagenetTransferLearning(LightningModule):
    # network input
    # Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        weight_path = r'D:\Users\lVavrek\research\data\sim-clr-backups\01112021-self-supervised-librispeech-epoch=19-val_loss=1.52.ckpt'
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
            x = x[0] # hack, because we use SimCLR transform which returns 3 augmented images; use custom tranform instead
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        x = self.sigmoid(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.celoss(logits, y) #F.nll_loss(logits, y)
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
        acc = FM.accuracy(y_hat, y)
        return loss, acc

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.0001) # , lr=0.001

# fine-tune
# model = ImagenetTransferLearning()
# trainer = Trainer()
# trainer.fit(model)

# prediction
# model = ImagenetTransferLearning.load_from_checkpoint(PATH)
# model.freeze()
# x = some_images_from_cifar10()
# predictions = model(x)

if __name__ == '__main__':
    train_self_supervised()
    # train_transfer_learning()

    exit()
    d1 = LibrispeechSpectrogramDataset(transform=SimCLRTrainDataTransform(input_height=224, gaussian_blur=False), train=True)
    d2 = PcGitaTorchDataset(transform=SimCLRTrainDataTransform(input_height=224, gaussian_blur=False), train=True)
    dl1 = DataLoader(d1, batch_size=16, num_workers=4)
    dl2 = DataLoader(d2, batch_size=16, num_workers=4)

    x1 = next(iter(dl1))

    print(len(x1))
    print(x1[1])
    print(len(x1[0]))

    x2 = next(iter(dl2))

    print(len(x2))
    print(x2[1])
    print(len(x2[0]))
    exit()

    x1 = next(iter(d1))

    print(len(x1))
    print(len(x2))

    print(len(x1[0]))
    print(len(x2[0]))

    print(len(x1[0][0][0]))
    print(len(x2[0][0][0]))
    print(len(x1[0][2][0]))
    print(len(x2[0][2][0]))

    test_loader = DataLoader(d2, batch_size=16, num_workers=4)

    net = ImagenetTransferLearning()
    x = next(iter(test_loader))[0][0][:1] # single sample, drop the label; TODO: check what is returned by data loader
    out = net(x)
    print(out)
    print('the end')
