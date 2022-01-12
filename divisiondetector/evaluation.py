import os
from pytorch_lightning.callbacks import Callback

class DivisionPerformanceValidation(Callback):

    def __init__(self):
        super().__init__()

    def on_validation_epoch_start(self, trainer, pl_module):
        """Called when the validation loop begins."""
        self.scores = {}
        self.out_dir = self.create_eval_dir(pl_module)

    def create_eval_dir(self, pl_module):
        eval_directory = os.path.abspath(os.path.join(pl_module.logger.log_dir,
                                                      os.pardir,
                                                      os.pardir,
                                                      "evaluation",
                                                      f"{pl_module.global_step:08d}"))

        os.makedirs(eval_directory, exist_ok=True)
        return eval_directory

    def on_validation_end(self, trainer, pl_module):
        eval_score_file = f"{self.out_dir}/scores_{pl_module.local_rank}.csv"
        scores = evaluate_predicted_zarr(self.output_file, eval_score_file)
        pl_module.metrics = scores.to_dict()

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # TODO: implement a validation procedure here
        # 1: predict with network
        # 2: apply non maximum supression
        # 3: measure recall precision and f1 score
        raise NotImplementedError()