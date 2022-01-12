# Note: COPY OF 'divisiondetector/train.py'

from argparse import ArgumentParser
from divisiondetector.datamodules import DivisionDataModule
from divisiondetector.trainingmodules import DivisionDetectorTrainer

import pytorch_lightning as pl


if __name__ == '__main__':
    
    parser = ArgumentParser()

    parser = DivisionDetectorTrainer.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DivisionDataModule.add_argparse_args(parser)
    parser = DivisionDataModule.add_model_specific_args(parser)

    args = parser.parse_args()

    model = DivisionDetectorTrainer.from_argparse_args(args)
    datamodule = DivisionDataModule.from_argparse_args(args)

    #  init trainer
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule)
