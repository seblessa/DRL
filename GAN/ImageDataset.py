from torch.utils.data import Dataset
from typing import Callable, Tuple
from torch import Tensor
from PIL import Image
import pandas as pd
import torch

class ImageDataset(Dataset):
    """
    A PyTorch Dataset class for loading images and corresponding labels
    from a pandas DataFrame.

    Attributes:
        df (pd.DataFrame): DataFrame containing image file paths and labels.
        transform (Callable): Transformation function to apply to images.
    """

    def __init__(self, df: pd.DataFrame, transform: Callable):
        """
        Initializes the ImageDataset.

        Args:
            df (pd.DataFrame): A DataFrame with at least two columns:
                               'path' (str) for image file paths and 
                               'label' (float/int) for labels.
            transform (Callable): A function or torchvision.transforms object 
                                  to apply to each image.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of items in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Retrieves the image and label at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the transformed image tensor 
                                   and the corresponding label tensor.
        """
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor([row["label"]], dtype=torch.float32)
        return img, label
