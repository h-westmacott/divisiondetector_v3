from divisiondetector.datasets import DivisionDataset
import pandas as pd
import numpy as np
import chardet

'''with open("processed_16-10-13-EGG_manual_annotations_t10_t103-t107_t186-t190-vertices.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

df = pd.read_csv("processed_16-10-13-EGG_manual_annotations_t10_t103-t107_t186-t190-vertices.csv", encoding=result['encoding'])

print(df["Spot frame"].describe())
print(len(df[df["Spot frame"]=="188"]))'''

dataset = DivisionDataset(
    "data/processed_16-10-13-EGG_manual_annotations_t10_t103-t107_t186-t190-vertices.csv",
    "data/volumes.zarr",
    "data/processed_16-10-13-EGG_manual_annotations_t10_t103-t107_t186-t190-vertices.csv",
    (500, 500, 500),
    (1, 1),
    'ball',
    (10, 10, 10)
)

print(len(dataset))

img, divs = dataset[5]
#print(img)
print(type(divs))