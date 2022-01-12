import os
import pandas as pd
import chardet
import glob
import logging

dest_folder = 'processed_data_for_gp'
origin_folder = 'processed_data'

logger = logging.getLogger()

def output_csv(origin_path, dest_path):
    '''
    Load and process label data
    '''
    with open(origin_path, 'rb') as f:
        result = chardet.detect(f.read(10000))

    raw_df = pd.read_csv(origin_path, encoding=result['encoding'])

    logger.info("Data loaded.")

    raw_df["ID"] = raw_df["ID"].astype(int) # Ensure the IDs are integers
    raw_df = raw_df.set_index("ID") # Set index to ID

    columns = ["Timepoint", "Z", "Y", "X"]
    df = raw_df.apply(lambda x: pd.Series([int(x[0])] + [float(element) for element in x[1:]], index=columns), axis=1) # Convert coordinates and timepoints to numbers (from strings) and relabel

    logger.info("Data processed.")
    
    # Writing the data to a CSV

    # Gunpowder wants its CSVs separated with ', ' and not just ','
    write_df = df.apply(lambda x: pd.Series([str(element) + ',' for element in x], index=columns), axis=1)
    write_df[columns].assign(id=write_df.index.to_series()).to_csv(dest_path, sep=' ', index=False, header=False)

    logger.info("Data written.")

if __name__ == '__main__':
    if not os.path.isdir(dest_folder):
        os.mkdir(dest_folder)

    os.chdir(origin_folder)
    file_list = glob.glob('*.csv')
    os.chdir('..')

    for file in file_list:
        output_csv(os.path.join(origin_folder, file), os.path.join(dest_folder, file))