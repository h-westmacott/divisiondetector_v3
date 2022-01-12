import os
import zarr
import numpy as np
from pytorch_lightning import Callback

from divisiondetector.utils.utils import BuildFromArgparse

write_path = 'sample_training_output'

class MonitorCallback(Callback, BuildFromArgparse):
    def __init__(self, feedback_epoch=20, feedback_batch=0):
        self.feedback_epoch = feedback_epoch
        self.feedback_batch = feedback_batch
        self.state = {'current_epoch': 0}

    def on_validation_epoch_end(self, trainer, pl_module):
        self.state['current_epoch'] += 1

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx) -> None:
        if self.state['current_epoch'] == self.feedback_epoch and batch_idx == self.feedback_batch:
            x, y = batch
            y_hat = pl_module(x)

            os.makedirs(write_path, exist_ok=True)
            fn = os.path.join(write_path, f"epoch_{self.state['current_epoch']}_batch_{batch_idx}.zarr")

            self.write_zarr(x, y, y_hat, fn)

    def write_zarr(self, x, y, y_hat, fn):
        zarr_store = zarr.open(fn, 'w')

        b, c, t, d, h, w = x.shape
        zarr_store.create_dataset(
            "volumes",
            data=x.cpu().detach().numpy(),
            chunks=(b, c, t, d//4, h//8, w//8),
            compression='lz4',
            overwrite=True
        )

        y = y[:, np.newaxis, :, :, :, :]
        b, c, t, d, h, w = y.shape
        zarr_store.create_dataset(
            "divisions",
            data=y.cpu().detach().numpy(),
            chunks=(b, c, t, d//4, h//8, w//8),
            compression='lz4',
            overwrite=True
        )

        b, c, t, d, h, w = y_hat.shape
        zarr_store.create_dataset(
            "predictions",
            data=y_hat.cpu().detach().numpy(),
            chunks=(b, c, t, d//4, h//8, w//8),
            compression='lz4',
            overwrite=True
        )
    
    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        return self.state.update(callback_state)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        return self.state.copy()

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--feedback_epoch', type=int, default=20)
        parser.add_argument('--feedback_batch', type=int, default=0)
        return parser