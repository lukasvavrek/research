from dataclasses import dataclass

@dataclass
class SIMClr:
    batch_size: int
    input_height: int
    num_workers: int
    dataset: str


@dataclass
class SIMClrTL(SIMClr):
    max_epochs: int


@dataclass
class Params:
    run_visualizations: bool
    run_simclr: bool
    run_data_extraction: bool
    run_transfer_learning: bool
    accelerator: str


@dataclass
class Paths:
    data: str


@dataclass
class SIMClrConfig:
    paths: Paths
    params: Params
    simclr: SIMClr
    tl_simclr: SIMClrTL
