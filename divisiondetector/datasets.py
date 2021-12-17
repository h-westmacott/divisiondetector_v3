import os
import logging
import pandas as pd
from torch.utils.data import Dataset
from utils.gp_pipeline import GPPipeline

logger = logging.getLogger()

class DivisionDataset(Dataset):
    def __init__(self, label_path, img_path, window_size=(100,100,100), time_window=(1, 1), mode='ball', ball_radius=(10, 10, 10)):
        def __getlabels(label_path, div_path):
            '''
            Load and process label data
            '''
            raw_df = pd.read_csv(label_path, encoding='unicode_escape')

            logger.info("Data loaded.")

            # Converting and formatting
            columns = ["Timepoint", "X", "Y", "Z"]

            raw_df = raw_df.loc[2:].drop(["Label", "Detection quality"], axis=1) # Dropping unnecessary columns
            raw_df["ID"] = raw_df["ID"].astype(int) # Ensure the IDs are integers
            raw_df = raw_df.set_index("ID") # Set index to ID

            df = raw_df.apply(lambda x: pd.Series([int(x[0])] + [float(element) for element in x[1:]], index=columns), axis=1) # Convert coordinates and timepoints to numbers (from strings) and relabel

            logger.info("Data processed.")

            # Gunpowder wants its CSVs separated with ', ' and not just ','
            write_df = df.apply(lambda x: pd.Series([str(element)+',' for element in x], index=columns), axis=1)
            columns = ["Timepoint", "Z", "Y", "X"] # Reorder columns
            write_df["Timepoint"] = write_df["Timepoint"].astype(int)
            write_df[columns].assign(id=write_df.index.to_series()).to_csv(div_path, sep=' ', index=False, header=False)

            logger.info("Data written.")

            return df

        def __getCSV(label_path, div_path):
            if os.path.isfile(div_path):
                df = pd.read_csv(div_path, sep='\, ', header=None, engine='python')
                df.columns = ["Timepoint", "Z", "Y", "X", "ID"]
                df = df.set_index("ID")

                logger.info("File exists. Data loaded.")

                return df
            else:
                return __getlabels(label_path, div_path)

        logger.info("Initialising...")

        div_path = "division.csv"

        self.labels = __getCSV(label_path, div_path)
        self.img_path = img_path
        self.window_size = window_size
        self.time_window = time_window


        self.pipeline = GPPipeline(img_path, div_path, mode, ball_radius)

        logger.info("Pipeline created.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        '''
        5D-array
        '''
        data = self.labels.loc[idx]

        c_t_vol, target, points = self.pipeline.fetch_data(
            (data["Z"], data["Y"], data["X"]),
            self.window_size,
            data["Timepoint"],
            self.time_window
        )

        return c_t_vol, target, points