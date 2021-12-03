from typing import Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as albu
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import random


@dataclass_json
@dataclass
class DataAugmentationConfig:
    rotation_range_deg: int = 5
    shift_range: float = 0.1
    zoom_range: float = 0.1
    horizontal_flip: bool = False
    vertical_flip: bool = False

    def get_transform(self) -> albu.Compose:

        train_transform = []
        if self.horizontal_flip:
            train_transform.append(albu.HorizontalFlip(p=0.5))
        if self.vertical_flip:
            train_transform.append(albu.VerticalFlip(p=0.5))
        train_transform.append(albu.ShiftScaleRotate(scale_limit=self.zoom_range,
                                                     rotate_limit=self.rotation_range_deg,
                                                     shift_limit=self.shift_range,
                                                     p=1,
                                                     border_mode=0))
        return albu.Compose(train_transform)


class CustomDataset(Dataset):
    def __init__(self,
                 images: np.ndarray,
                 labels: np.ndarray,
                 class_count: int,
                 augmentation_config: Optional[DataAugmentationConfig] = None):
        assert labels.shape[1] == 2, 'Dataset works only for 2 digits.'
        assert len(labels) == len(images), 'Labels and images should have the same lenght.'
        self.images = images
        self.class_count = class_count
        self.labels = labels
        self._augmentation = augmentation_config.get_transform() if augmentation_config is not None else None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[idx]
        labels = self.labels[idx]

        if self._augmentation:
            # random label swap to enforce permutation invariance
            if random.uniform(0, 1) > 0.5:
                labels = labels[[1, 0]]
            # image augmentation
            image = self._augmentation(image=image)['image']

        image = torch.unsqueeze(torch.from_numpy(image).float(), 0)
        labels = torch.from_numpy(labels).long()
        return image, labels
