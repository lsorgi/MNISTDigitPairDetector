import datetime
import logging
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader
from copy import copy
from typing import Optional, Dict
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from enum import Enum
import contextlib
import json
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

from digit_pair_detector.dataset import DataAugmentationConfig
from digit_pair_detector.model import ModelConfig, MyDigitPairDetector
from digit_pair_detector.dataset import CustomDataset


@dataclass_json
@dataclass
class LearningRateSchedulerConfig:
    # for the list of supported classes we refer to https://pytorch.org/docs/stable/optim.html
    name: str = ''
    params: Dict[str, any] = field(default_factory=dict)


@dataclass_json
@dataclass
class OptimizerConfig:
    # for the list of supported classes we refer to https://pytorch.org/docs/stable/optim.html
    name: str = 'SGD'
    params: Dict[str, any] = field(default_factory=lambda: {'lr': 1e-4, 'momentum': 0.8})


@dataclass_json
@dataclass
class TrainingConfig:
    shuffle: bool = True
    batch_size: int = 20
    epochs: int = 50
    num_workers: int = 8
    device: str = 'cuda'
    random_seed: Optional[int] = None
    learning_rate_scheduler: LearningRateSchedulerConfig = LearningRateSchedulerConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    data_augmentation: DataAugmentationConfig = DataAugmentationConfig()


@dataclass_json
@dataclass
class TrainerConfig:
    images_file: str
    labels_file: str
    train_ratio: float = 0.7
    cnn_model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()

    @staticmethod
    def from_file(cfg_file: str) -> "TrainerConfig":
        with open(cfg_file, 'r') as fp:
            cfg = json.load(fp)
        return TrainerConfig.from_dict(cfg)

    def to_file(self, cfg_file: str):
        json_data = self.to_dict()
        with open(cfg_file, 'w') as fp:
            json.dump(json_data, fp, indent=4)
        logging.info(f'Configuration saved to file ({cfg_file})')


class EpochType(str, Enum):
    TRAIN = 'train'
    VALID = 'valid'


class Trainer:

    def __init__(self,
                 cfg: TrainerConfig,
                 output_folder: str,
                 job_name: Optional[str] = None):

        self.config = copy(cfg)

        # set random seed
        self._seed = self.config.training.random_seed if self.config.training.random_seed is not None \
            else int(datetime.datetime.now().microsecond)
        np.random.seed(self._seed)
        random.seed(self._seed)
        torch.manual_seed(self._seed)

        # get model
        self._model = MyDigitPairDetector(cfg=self.config.cnn_model)
        self._model.enable_probabilities(False)
        self._model = self._model.to(self.config.training.device)
        self._model.train()

        # set training- and validation-dataloader
        images = np.load(self.config.images_file)
        labels = np.load(self.config.labels_file)
        n_samples = len(images)
        indices = np.array(random.sample(range(n_samples), n_samples))
        train_len = int(self.config.train_ratio * n_samples)
        train_dataset = CustomDataset(images=images[indices[:train_len]],
                                      labels=labels[indices[:train_len]],
                                      class_count=self.config.cnn_model.class_count,
                                      augmentation_config=None)
        valid_dataset = CustomDataset(images=images[indices[train_len:]],
                                      labels=labels[indices[train_len:]],
                                      class_count=self.config.cnn_model.class_count,
                                      augmentation_config=None)
        self.dataloaders = {
            EpochType.TRAIN: DataLoader(dataset=train_dataset,
                                        batch_size=cfg.training.batch_size,
                                        shuffle=cfg.training.shuffle,
                                        num_workers=cfg.training.num_workers),
            EpochType.VALID: DataLoader(dataset=valid_dataset,
                                        batch_size=cfg.training.batch_size,
                                        shuffle=False,
                                        num_workers=cfg.training.num_workers)
        }
        logging.info(f'Dataloaders initialized')
        logging.info(f'training samples ({train_len}), '
                     f'training batches ({len(self.dataloaders[EpochType.TRAIN])})')
        logging.info(f'validation samples ({n_samples - train_len}), '
                     f'training batches ({len(self.dataloaders[EpochType.VALID])})')

        # set loss
        self._loss = torch.nn.CrossEntropyLoss(reduction ='none')

        # get optimizer
        klass = getattr(torch.optim, self.config.training.optimizer.name)
        self.optimizer = klass(params=self._model.parameters(),
                               **self.config.training.optimizer.params)

        # learning rate scheduler
        if self.config.training.learning_rate_scheduler.name:
            klass = getattr(torch.optim.lr_scheduler, self.config.training.learning_rate_scheduler.name)
            self.lr_scheduler = klass(optimizer=self.optimizer,
                                      **self.config.training.learning_rate_scheduler.params)
        else:
            self.lr_scheduler = None

        # set output destination_folder tree
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        job_name = f'{job_name}_{timestamp}' if job_name else f'test_{timestamp}'
        self.output_dir = Path(output_folder).expanduser() / job_name
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        logging.info(f'Output Folder: {job_name}')
        self.config.to_file((self.output_dir / 'config.json').as_posix())

        # set up tensorboard logger
        self.log_dir = self.output_dir / 'logs'
        self.log_dir.mkdir()
        self.writer = SummaryWriter(log_dir=self.log_dir.as_posix())

    def run(self) -> None:
        """
        Runs the optimization loop for `self.epochs` and saves the model to output destination_folder

        """
        best_validation_loss = np.inf
        for epoch in range(self.config.training.epochs):
            logging.info('\nEpoch: {}'.format(epoch))

            self._epoch(epoch, EpochType.TRAIN)
            valid_loss = self._epoch(epoch, EpochType.VALID)

            if valid_loss < best_validation_loss:
                model_path = self.checkpoint_dir / f'best_model.pth'
                torch.save(self._model, str(model_path))
                logging.info(f'Checkpoint after {epoch} epochs saved!')
                best_validation_loss = valid_loss

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(valid_loss)

        model_path = self.checkpoint_dir / 'model.pth'
        torch.save(self._model, str(model_path))
        logging.info('Model saved!')
        self.writer.close()

    def _epoch(self,
               epoch_index: int,
               epoc_type: EpochType) -> float:

        # Set epoch mode
        if epoc_type == EpochType.TRAIN:
            context = contextlib.nullcontext
            self._model.train()
        else:
            context = torch.no_grad
            self._model.eval()

        # run epoch
        with context():

            # Step through dataset
            epoch_samples = 0
            epoch_loss = 0
            epoch_correct_preds = 0

            for bn, batch_data in enumerate(tqdm(self.dataloaders[epoc_type])):

                image_batch = batch_data[0].to(self.config.training.device)
                label_batch = batch_data[1].to(self.config.training.device)

                batch_samples = len(image_batch)
                epoch_samples += batch_samples

                digits0 = label_batch[:, 0]
                digits1 = label_batch[:, 1]

                # forward pass
                pred_batch = self._model.forward(image_batch)
                pred_scores0 = pred_batch[0]
                pred_scores1 = pred_batch[1]

                # compute loss
                loss_a = self._loss.forward(pred_scores0, digits0).to(self.config.training.device) + \
                         self._loss.forward(pred_scores1, digits1).to(self.config.training.device)
                loss_b = self._loss.forward(pred_scores0, digits1).to(self.config.training.device) + \
                         self._loss.forward(pred_scores1, digits0).to(self.config.training.device)
                loss = torch.sum(torch.minimum(loss_a, loss_b))

                # update metrics
                epoch_loss += loss.item()

                digits0 = digits0.detach().cpu().numpy()
                digits1 = digits1.detach().cpu().numpy()
                pred_digits0 = torch.argmax(pred_scores0, dim=1).detach().cpu().numpy()
                pred_digits1 = torch.argmax(pred_scores1, dim=1).detach().cpu().numpy()
                flags = np.logical_or(np.logical_and(digits0 == pred_digits0, digits1 == pred_digits1),
                                      np.logical_and(digits1 == pred_digits0, digits0 == pred_digits1))
                epoch_correct_preds += np.sum(flags)

                # backward pass
                if epoc_type == EpochType.TRAIN:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # compute epoch metrics
            epoch_loss = epoch_loss / len(self.dataloaders[epoc_type])
            epoch_accuracy = epoch_correct_preds / epoch_samples
            logging.info(f'Epoch({epoch_index}, {epoc_type}), loss({epoch_loss}), accuracy({epoch_accuracy})')

            self.writer.add_scalar(tag=f'loss / {epoc_type} (epoch)',
                                   scalar_value=epoch_loss,
                                   global_step=epoch_index)
            self.writer.add_scalar(tag=f'accuracy / {epoc_type} (epoch)',
                                   scalar_value=epoch_accuracy,
                                   global_step=epoch_index)

        return float(epoch_loss)
