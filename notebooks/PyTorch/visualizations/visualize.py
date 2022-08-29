import torchvision
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from pytorch_lightning.loggers import TensorBoardLogger

from librispeech.dataset import LibrispeechSpectrogramDataset
from notebooks.PyTorch.config.simcrlplaygroundconfig import SIMClrConfig
from pcgita.dataset import PcGitaTorchDataset


def visualize_spectrograms_librispeech(cfg: SIMClrConfig) -> None:
    logger = TensorBoardLogger('runs', name='SimCLR_libri_speech')

    td = LibrispeechSpectrogramDataset(transform=None, train=True, root=cfg.paths.data)

    grid = make_img_grid(td, 24)

    logger.experiment.add_image("generated_images", grid, 0)
    logger.finalize("success")

def make_img_grid(dataset: Dataset, count: int) -> Tensor:
    to_tensor = transforms.ToTensor()

    samples = []
    for i in range(0, count):
        img, cls = dataset.__getitem__(i)
        img = to_tensor(img)
        samples.append(img)

    return torchvision.utils.make_grid(samples, padding=10, nrow=4)


def visualize_spectrograms_pcgita(cfg: SIMClrConfig):
    logger = TensorBoardLogger('runs', name='pc-gita')

    td = PcGitaTorchDataset(transform=None, train=True, root=cfg.paths.data)

    grid = make_img_grid(td, 24)

    logger.experiment.add_image("generated_images", grid, 0)
    logger.finalize("success")
