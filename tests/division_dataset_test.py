from divisiondetector.datasets import DivisionDataset
import pandas as pd
import numpy as np
import chardet

with open("16-10-13-EGG_manual_annotations_t10_t103-t107_t186-t190-vertices.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

print(result)

df = pd.read_csv("16-10-13-EGG_manual_annotations_t10_t103-t107_t186-t190-vertices.csv", encoding=result['encoding'])

#print(df["Spot frame"].describe())
#print(len(df[df["Spot frame"]=="188"]))

timepoints = [186, 187, 188, 189, 190] # Sorted and indexed!
dataset = DivisionDataset(
    "16-10-13-EGG_manual_annotations_t10_t103-t107_t186-t190-vertices.csv",
    "volumes.zarr",
    (500, 500, 500),
    (1, 1),
    'ball',
    (10, 10, 10)
)

print(len(dataset))

img, divs, pts = dataset[77117] # Handpicked for Spot frame/Timepoint 188
#print(img)
#print(divs)
#print(pts)