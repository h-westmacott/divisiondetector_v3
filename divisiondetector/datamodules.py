import os
import json
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from divisiondetector.datasets import DivisionDataset

class DivisionDataModule(pl.LightningDataModule):

    def __init__(self,
                 config_path,
                 window_size,
                 batch_size,
                 loader_workers=10):
        super().__init__()

        self.config = json.loads(config_path) # Dictionary
        self.window_size = tuple(window_size)
        self.batch_size = batch_size
        self.loader_workers = loader_workers

    def setup(self, stage=None):
        window_size = self.window_size
        time_window = (1, 1)
        mode = 'ball'
        ball_radius = (10, 10, 10)

        def create_dataset(path_list):
            datasets = []
            for paths in path_list:
                datasets.append(DivisionDataset(
                    paths['data'],
                    paths['volume'],
                    paths['labels'],
                    window_size,
                    time_window,
                    mode,
                    ball_radius
                ))
            return ConcatDataset(datasets)

        self.ds_train = create_dataset(self.config['train'])
        self.ds_val = create_dataset(self.config['validation'])
        self.ds_test = create_dataset(self.config['test'])

    def train_dataloader(self):
        return DataLoader(self.ds_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.loader_workers)

    def val_dataloader(self):
        return DataLoader(self.ds_val,
                          batch_size=self.batch_size,
                          num_workers=2,
                          drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.ds_test,
                          batch_size=self.batch_size,
                          num_workers=2,
                          drop_last=False)

    @ staticmethod
    def add_model_specific_args(parent_parser):
        # add all arguments from the __init__ function here
        # it can then be build using the BuildFromArgparse class mixin
        # see train.py for an example

        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        try:
            parser.add_argument('--batch_size', type=int, default=8)
        except argparse.ArgumentError:
            pass
        parser.add_argument('--loader_workers', type=int, default=8)
        parser.add_argument('--window_size', nargs=3, type=int, default=500) # (z, y, x)
        parser.add_argument('--config_path', type=str)
        return parser
