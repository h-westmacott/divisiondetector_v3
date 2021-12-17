from argparse import ArgumentParser
import inspect
import os
import torch
from pytorch_lightning.callbacks import Callback

class BuildFromArgparse(object):
    @classmethod
    def from_argparse_args(cls, args, **kwargs):

        if isinstance(args, ArgumentParser):
            args = cls.parse_argparser(args)
        params = vars(args)

        # we only want to pass in valid DataModule args, the rest may be user specific
        valid_kwargs = inspect.signature(cls.__init__).parameters
        datamodule_kwargs = dict(
            (name, params[name]) for name in valid_kwargs if name in params
        )
        datamodule_kwargs.update(**kwargs)

        return cls(**datamodule_kwargs)


class SaveModelOnValidation(Callback):

    def __init__(self, device='cpu'):
        self.device = device
        super().__init__()

    def on_validation_epoch_start(self, trainer, pl_module):
        """Called when the validation loop begins."""
        model_directory = os.path.abspath(os.path.join(pl_module.logger.log_dir,
                                                       os.pardir,
                                                       os.pardir,
                                                       "models"))
        os.makedirs(model_directory, exist_ok=True)
        if hasattr(pl_module, "unet"):
            model_save_path = os.path.join(
                model_directory, f"unet_{pl_module.global_step:08d}_{pl_module.local_rank:02}.torch")
            torch.save({"model_state_dict": pl_module.unet.state_dict()}, model_save_path)
