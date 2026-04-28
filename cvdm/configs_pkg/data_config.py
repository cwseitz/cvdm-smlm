from dataclasses import dataclass


@dataclass
class DataConfig:
    """
    Configuration for the dataset.

    Attributes:
        dataset_path (str): Path to the dataset directory used by the SMLM dataloader.
        n_samples (int): Number of samples to use from the dataset.
        batch_size (int): Number of samples per batch during training.
        im_size (int): The size of the patches of images (both height and width) to use.
    """

    dataset_path: str
    n_samples: int
    batch_size: int
    im_size: int
