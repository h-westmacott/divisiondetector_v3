import pandas as pd
import glob
import chardet

def output_csv(video_str, annotated_frames):
    # Manually annotated
    annotated = []

    for file in glob.glob(video_str[:2] + '-'+ video_str[2:4] + '-'+ video_str[4:] + '-*vertices.csv'):
        frames = annotated_frames[file]

        with open(file, 'rb') as f:
            result = chardet.detect(f.read(10000))
        
        # Reading CSV
        df = pd.read_csv(file, encoding=result['encoding'])[2:]

        # Filtering annotated frames
        df['Spot frame'] = df['Spot frame'].astype(int)
        df = df[df['Spot frame'].isin(frames)]

        annotated.append(df)

    # Concatenate all DataFrames
    annotated_df = pd.concat(annotated)

    annotated_df = annotated_df.rename(columns={
        'Spot position': 'X',
        'Spot position.1': 'Y',
        'Spot position.2': 'Z'
    }).reset_index(drop=True)

    # Make data floats
    annotated_df[['X', 'Y', 'Z']] = annotated_df[['X', 'Y', 'Z']].apply(lambda x:
        pd.Series([float(element) for element in x]))


    # Predicted
    predicted = []

    for file in glob.glob('*' + video_str + '_0.3.csv'):
        with open(file, 'rb') as f:
            result = chardet.detect(f.read(10000))

        # Read CSV
        headers = ['Z', 'Y', 'X', 'Spot frame', 'Detection Quality']
        predicted.append(pd.read_csv(file, encoding=result['encoding'], names=headers))

    # Concatenate all DataFrames
    predicted_df = pd.concat(predicted)

    # Filter annotated frames
    predicted_df = predicted_df[predicted_df['Spot frame'].isin(sum(annotated_frames.values(), []))]

    # Remove true annotations
    negative_df = predicted_df[predicted_df.apply(lambda x: 
        annotated_df.loc[
            (annotated_df['X'] == x['X']) &
            (annotated_df['Y'] == x['Y']) &
            (annotated_df['Z'] == x['Z'])].empty, axis=1)]

    annotated_df = annotated_df[['Spot frame', 'Z', 'Y', 'X']].rename(columns={'Spot frame': 'Timepoint'})
    negative_df = negative_df[['Spot frame', 'Z', 'Y', 'X']].rename(columns={'Spot frame': 'Timepoint'})
    predicted_df = predicted_df[['Spot frame', 'Z', 'Y', 'X']].rename(columns={'Spot frame': 'Timepoint'})

    annotated_df.to_csv('sorted_data/positive_divisions_' + video_str + '.csv', index_label='ID')
    negative_df.to_csv('sorted_data/negative_divisions_' + video_str + '.csv', index_label='ID')
    predicted_df.to_csv('sorted_data/all_divisions_' + video_str + '.csv', index_label='ID')


videos = [161013, 160616]

annotated_frames = {
    161013: {
        '16-10-13-EGG_manual_annotations_t10_t103-t107_t186-t190-vertices.csv': [
            10, 103, 104, 105, 106, 107, 186, 187, 188, 189, 190
        ],
        '16-10-13-EGG_manual_annotations_t29-t33_t59-vertices.csv': [
            29, 30, 31, 32, 33, 59
        ]
    },
    160616: {
        '16-06-16-EGG-manual_annotations_t10-t14_t135-t139_t281-t285-vertices.csv': [
            10, 11, 12, 13, 14, 135, 136, 137, 138, 139, 281, 282, 283, 284, 285
        ]
    }
}

for video in videos:
    output_csv(str(video), annotated_frames[video])