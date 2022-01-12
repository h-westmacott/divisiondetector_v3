import os
import logging
import chardet
import pandas as pd
from torch.utils.data import Dataset
from .utils.gp_pipeline import GPPipeline

logger = logging.getLogger()

class DivisionDataset(Dataset):
    def __init__(self, all_labels_path, vol_path, truth_labels_path, window_size=(100,100,100), time_window=(1, 1), mode='ball', ball_radius=(10, 10, 10)):
        '''
        Initialising the Division Dataset

        all_labels_path: For the model to iterate through while training/testing/validating. Can be both positive or
        negative examples.

        vol_path: The path to the Zarr file containing the light sheet video.

        truth_labels_path: Ground truth for the video at vol_path. Includes only POSITIVE examples.

        window_size: Size of the query window for data. Formatted (z, y, x)

        time_window: Determines how many frames before and after the specified timepoint the pipeline should
        return when querying.

        mode: Mode of the point rasterisation. Can be either 'ball' or 'peak'

        ball_radius: Radius of rasterised ball.
        '''

        def __getCSV(label_path):
            if os.path.isfile(label_path):
                with open(label_path, 'rb') as f:
                    result = chardet.detect(f.read(10000))
                df = pd.read_csv(label_path, sep='\, ', header=None, encoding=result['encoding'], engine='python')
                df.columns = ["Timepoint", "Z", "Y", "X", "ID"]
                df = df.reset_index()

                logger.info("File exists. Data loaded.")

                return df
            else:
                raise FileNotFoundError('File not found at label_path')

        logger.info("Initialising...")

        self.labels = __getCSV(all_labels_path) # Data to train/validate/test with (positive and negative)
        self.window_size = window_size
        self.time_window = time_window

        self.pipeline = GPPipeline(vol_path, truth_labels_path, mode, ball_radius) # Data to train/validate/test on (positive only)

        logger.info("Pipeline created.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        '''
        5D-array
        '''
        label = self.labels.loc[idx]

        c_t_vol, target, points = self.pipeline.fetch_data(
            (label["Z"], label["Y"], label["X"]),
            self.window_size,
            label["Timepoint"],
            self.time_window
        )

        return c_t_vol, target