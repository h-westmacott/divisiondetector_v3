from argparse import ArgumentParser
from divisiondetector.datamodules import DivisionDataModule
from divisiondetector.trainingmodules import DivisionDetectorTrainer
from divisiondetector.utils.utils import SaveModelOnValidation
from divisiondetector.monitoring import MonitorCallback
# from divisiondetector.evaluation import DivisionPerformanceValidation

import pytorch_lightning as pl


if __name__ == '__main__':

    parser = ArgumentParser()

    parser = DivisionDetectorTrainer.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DivisionDataModule.add_argparse_args(parser)
    parser = DivisionDataModule.add_model_specific_args(parser)
    parser = MonitorCallback.add_argparse_args(parser)

    args = parser.parse_args()

    model = DivisionDetectorTrainer.from_argparse_args(args)
    datamodule = DivisionDataModule.from_argparse_args(args)
    # TODO: (when we have a working training module) Add a validation callback here
    # Ignore that for now :)
    # div_val = DivisionPerformanceValidation()
    monitor_callback = MonitorCallback.from_argparse_args(args)
    model_saver = SaveModelOnValidation()

    #  init trainer
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.callbacks.append(model_saver)
    trainer.callbacks.append(monitor_callback)
    # trainer.callbacks.append(div_val)
    trainer.fit(model, datamodule)
