# https://pytorch.org/audio/stable/_modules/torchaudio/datasets/librispeech.html#LIBRISPEECH

# this script is pretty much a copy of SIM CLR playground.ipynb
# I created it to try to avoid those bugs:
# https://github.com/PyTorchLightning/lightning-bolts/issues/640
# https://github.com/PyTorchLightning/lightning-bolts/issues/642

# https://github.com/PyTorchLightning/Lightning-Bolts/blob/master/pl_bolts/models/self_supervised/simclr/transforms.py
# https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.loggers.TensorBoardLogger.html#pytorch_lightning.loggers.TensorBoardLogger

import os
import sys

import hydra
from hydra.core.config_store import ConfigStore
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader

from librispeech.dataset import LibrispeechSpectrogramDataset
from notebooks.PyTorch.config.simcrlplaygroundconfig import SIMClrConfig
from notebooks.PyTorch.imagenettransferlearning import ImagenetTransferLearning
from notebooks.PyTorch.visualizations.visualize import visualize_spectrograms_librispeech, visualize_spectrograms_pcgita
from pcgita.dataset import PcGitaTorchDataset

# fixing imports TODO: fix this
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def train_self_supervised(cfg: SIMClrConfig):
    logger = TensorBoardLogger('runs', name='SimCLR_libri_speech')

    train_dataset = LibrispeechSpectrogramDataset(
        root=cfg.paths.data,
        transform=SimCLRTrainDataTransform(input_height=cfg.simclr.input_height, gaussian_blur=False),
        train=True)
    val_dataset = LibrispeechSpectrogramDataset(
        root=cfg.paths.data,
        transform=SimCLREvalDataTransform(input_height=cfg.simclr.input_height, gaussian_blur=False),
        train=False)

    train_loader = DataLoader(train_dataset, batch_size=cfg.simclr.batch_size, num_workers=cfg.simclr.num_workers)
    test_loader = DataLoader(val_dataset, batch_size=cfg.simclr.batch_size, num_workers=cfg.simclr.num_workers)

    model = SimCLR(gpus=1, num_samples=len(train_dataset), batch_size=cfg.simclr.batch_size, dataset=cfg.simclr.dataset)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=cfg.paths.data,
        filename="self-supervised-librispeech-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    early_stopping = EarlyStopping(monitor="val_loss")

    trainer = Trainer(
        gpus=1,
        accelerator=cfg.params.accelerator,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger)
    trainer.fit(model, train_loader, test_loader)


def train_transfer_learning(cfg: SIMClrConfig):
    logger = TensorBoardLogger('runs', name='pc-gita')

    train_dataset = PcGitaTorchDataset(
        root=cfg.paths.data,
        transform=SimCLRTrainDataTransform(input_height=cfg.tl_simclr.input_height, gaussian_blur=False),
        train=True)
    val_dataset = PcGitaTorchDataset(
        root=cfg.paths.data,
        transform=SimCLRTrainDataTransform(input_height=cfg.tl_simclr.input_height, gaussian_blur=False),
        train=False)

    train_loader = DataLoader(train_dataset, batch_size=cfg.tl_simclr.batch_size, num_workers=cfg.tl_simclr.num_workers)
    test_loader = DataLoader(val_dataset, batch_size=cfg.tl_simclr.batch_size, num_workers=cfg.tl_simclr.num_workers)

    model = ImagenetTransferLearning()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=cfg.paths.data,
        filename="transfer-learning-pcgita-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    trainer = Trainer(
        gpus=1,
        accelerator=cfg.params.accelerator,
        callbacks=[checkpoint_callback],
        logger=logger,
        max_epochs=cfg.tl_simclr.max_epochs)
    trainer.fit(model, train_loader, test_loader)


# new 3.4.
def extract_data_from_pretrained_model():
    logger = TensorBoardLogger('runs', name='pc-gita')

    batch_size = 32
    input_height = 224
    num_workers = 4

    train_dataset = PcGitaTorchDataset(
        transform=SimCLRTrainDataTransform(input_height=input_height, gaussian_blur=False), train=True)
    val_dataset = PcGitaTorchDataset(transform=SimCLRTrainDataTransform(input_height=input_height, gaussian_blur=False),
                                     train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    print('train_dataset: {}'.format(len(train_dataset)))
    print('val_dataset: {}'.format(len(val_dataset)))

    model = ImagenetTransferLearning(extract_features=True)

    for step, (x, y) in enumerate(train_loader):
        print('step {}'.format(step))
        print('[x] type: {} len: {}'.format(type(x), len(x)))
        print('[x0] type: {} size: {}'.format(type(x[0]), x[0].size()))
        print('[x1] type: {} size: {}'.format(type(x[1]), x[1].size()))
        print('[x2] type: {} size: {}'.format(type(x[2]), x[2].size()))
        print('[y] type: {} size: {}'.format(type(y), y.size()))

        y_pred = model(x)
        print('y_pred type: {} size: {}'.format(type(y_pred), y_pred.size()))

        y_np = y.detach().cpu().numpy()
        print('[y_np] type: {} size: {}'.format(type(y_np), y_np.shape()))

        y_pred_np = y_pred.detach().cpu().numpy()
        print('y_pred_np type: {} shape: {}'.format(type(y_pred_np), y_pred_np.shape))


# fine-tune
# model = ImagenetTransferLearning()
# trainer = Trainer()
# trainer.fit(model)

# prediction
# model = ImagenetTransferLearning.load_from_checkpoint(PATH)
# model.freeze()
# x = some_images_from_cifar10()
# predictions = model(x)


# Important, in order to use the strong types as a parameter with Hydra
cs = ConfigStore.instance()
cs.store(name="simclr_config", node=SIMClrConfig)


@hydra.main(version_base=None, config_path="config", config_name="simclrplayground")
def main(cfg: SIMClrConfig) -> None:
    if cfg.params.run_visualizations:
        print("Visualizing spectrograms...")
        visualize_spectrograms_librispeech(cfg)
        visualize_spectrograms_pcgita(cfg)

    if cfg.params.run_simclr:
        print("Training SIMClr self supervised network...")
        train_self_supervised(cfg)

    if cfg.params.run_transfer_learning:
        print("Training transfer learning model...")
        train_transfer_learning()

    if cfg.params.run_data_extraction:
        print("Extracting data from SIMClr pre-trained model...")
        extract_data_from_pretrained_model()

    print("Finished!")


if __name__ == '__main__':
    main()
    exit(0)
