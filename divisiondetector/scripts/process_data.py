import os
import pandas as pd
import chardet

dest_folder = 'processed_data'
origin_folder = 'sorted_data'

fname_160616 = os.path.join(origin_folder, 'all_divisions_160616.csv')
fname_161013 = os.path.join(origin_folder, 'all_divisions_161013.csv')

with open(fname_160616, 'rb') as f:
    result = chardet.detect(f.read(10000))

df_160616 = pd.read_csv(fname_160616, encoding=result['encoding'], index_col='ID')
df_161013 = pd.read_csv(fname_161013, encoding=result['encoding'], index_col='ID')

'''
16-10-13 frame t59 as a validation frame
test on 16-06-16 frames t135-t139
the rest is training data.
'''

val_df = df_161013[df_161013['Timepoint']==59]
test_df = df_160616[df_160616['Timepoint'].isin(range(135, 139+1))]

train_df_161013 = df_161013[df_161013['Timepoint']!=59]
train_df_160616 = df_160616[~df_160616['Timepoint'].isin(range(135, 139+1))]

if not os.path.exists(dest_folder):
    os.mkdir(dest_folder)

val_df.to_csv(os.path.join(dest_folder, 'validation_161013.csv'))
test_df.to_csv(os.path.join(dest_folder, 'test_160616.csv'))
train_df_161013.to_csv(os.path.join(dest_folder, 'train_161013.csv'))
train_df_160616.to_csv(os.path.join(dest_folder, 'train_160616.csv'))