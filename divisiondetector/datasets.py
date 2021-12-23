import os
import logging
import pandas as pd
from torch.utils.data import Dataset
from utils.gp_pipeline import GPPipeline

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
        '''

        def __getlabels(label_path, div_path=None):
            '''
            Load and process label data
            '''
            raw_df = pd.read_csv(label_path, encoding='unicode_escape')

            logger.info("Data loaded.")

            # Converting and formatting
            columns = ["Timepoint", "X", "Y", "Z"]

            try:
                raw_df = raw_df.drop(["Label", "Detection quality"], axis=1, errors='ignore') # Dropping unnecessary columns
                raw_df["ID"] = raw_df["ID"].astype(int) # Ensure the IDs are integers
            except:
                # In case of NaNs
                raw_df = raw_df.loc[2:].drop(["Label", "Detection quality"], axis=1, errors='ignore') # Dropping unnecessary columns
                raw_df["ID"] = raw_df["ID"].astype(int) # Ensure the IDs are integers

            raw_df = raw_df.set_index("ID") # Set index to ID

            # Reordering the columns in preparation for processing. Takes into account all the possible pre-processing done
            if 'X' in raw_df.columns and 'Timepoint' in raw_df.columns:
                raw_df = raw_df[columns]
            elif 'X' in raw_df.columns and 'Spot frame' in raw_df.columns:
                raw_df = raw_df[['Spot frame', 'X', 'Y', 'Z']]
            else: # Completely unprocessed
                raw_df = raw_df[['Spot frame', 'Spot position', 'Spot position.1', 'Spot position.2']]

            df = raw_df.apply(lambda x: pd.Series([int(x[0])] + [float(element) for element in x[1:]], index=columns), axis=1) # Convert coordinates and timepoints to numbers (from strings) and relabel

            logger.info("Data processed.")
            
            # Writing the data to a CSV (for convenience) if a path to write to is provided
            if div_path != None:
                # Gunpowder wants its CSVs separated with ', ' and not just ','
                write_df = df.apply(lambda x: pd.Series([str(element) + ',' for element in x], index=columns), axis=1)
                columns = ["Timepoint", "Z", "Y", "X"] # Reorder columns
                write_df[columns].assign(id=write_df.index.to_series()).to_csv(div_path, sep=' ', index=False, header=False)

                logger.info("Data written.")

            return df

        def __getCSV(label_path, div_path=None):
            # If a pre-processed convenience data file is present at specified path, read it
            if div_path != None:
                if os.path.isfile(div_path):
                    df = pd.read_csv(div_path, sep='\, ', header=None, engine='python')
                    df.columns = ["Timepoint", "Z", "Y", "X", "ID"]
                    df = df.set_index("ID")

                    logger.info("File exists. Data loaded.")

                    return df
            return __getlabels(label_path, div_path)

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

        return c_t_vol, target, points