import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import Unet4D
import torch
import torch.nn as nn
import numpy as np
import gunpowder as gp
import zarr
# from utils.div_unet import DivUNet
# from funlib.learn.torch.models.conv4d import Conv4d

def predict(predictconfig):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet4D(in_channels=1,out_channels=1)
    
    model= nn.DataParallel(model)
    # device = torch.device('cuda:1')
    model = model.to(device)

    checkpoint = predictconfig["checkpoint"]
    print(f"Resuming model from {checkpoint}")
    state = torch.load(checkpoint, map_location=device)
    # start_iteration = state["iteration"] + 1
    # lowest_loss = state["lowest_loss"]
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    # optimizer.load_state_dict(state["optim_state_dict"])
    # logger.data = state["logger_data"]

    input_shape = gp.Coordinate(
        (predictconfig["num_channels"], *predictconfig["crop_size"])
    )

    output_shape = gp.Coordinate(
        model(
            torch.zeros(
                (1, predictconfig["num_channels"], *predictconfig["crop_size"]),
                dtype=torch.float32,
            ).to(device)
        ).shape
    )

    voxel_size = (1,) * predictconfig["num_dims"]
    raw_spec = gp.ArraySpec(voxel_size=voxel_size[:5], interpolatable=True)

    input_size = gp.Coordinate((1,)+input_shape) * gp.Coordinate(voxel_size)
    output_size = gp.Coordinate(output_shape) * gp.Coordinate(voxel_size)
    diff_size = input_size - output_size
    context = (
            # 0,
            0,
            diff_size[2] // 2,
            diff_size[3] // 2,
            diff_size[4] // 2,
            diff_size[5] // 2,
            )  # type: ignore
    
    raw = gp.ArrayKey("RAW")
    prediction = gp.ArrayKey("PREDICT")

    scan_request = gp.BatchRequest()

    scan_request[raw] = gp.Roi(
        (0, -diff_size[2] // 2, -diff_size[3] // 2, -diff_size[4] // 2, -diff_size[5] // 2),
        (
            predictconfig["num_channels"],
            input_size[2],
            input_size[3],
            input_size[4],
            input_size[5],
        ),
    )
    scan_request[prediction] = gp.Roi(
        (0, 0, 0, 0, 0),
        (
            # 1,
            1,
            output_size[2],
            output_size[3],
            output_size[4],
            output_size[5],
        ),
    )

    predict = gp.torch.Predict(
        model,
        inputs={"raw": raw},
        outputs={0: prediction},
        array_specs={prediction: raw_spec},
    )

    container = zarr.open(predictconfig["raw_container_path"])
    data = container[predictconfig["raw_dataset_name"]]
    spatial_array = data.shape[-4:]
    # prepare the zarr dataset to write to
    f = zarr.open(predictconfig["prediction_container_path"])
    ds = f.create_dataset(
        predictconfig["prediction_dataset_name"],
        shape=(
            # 1,
            1,
            *spatial_array,
        ),
        dtype=float,
        overwrite=True
    )

    class Print(gp.BatchFilter):
        def __init__(self, prefix):
            self.prefix = prefix

        def prepare(self, request):
            print(f"{self.prefix}\tRequest going upstream: {request}")

        def process(self, batch, request):
            print(f"{self.prefix}\tBatch going downstream: {batch}")

    pipeline = (
        gp.ZarrSource(
            predictconfig["raw_container_path"],
            {raw: predictconfig["raw_dataset_name"]},
            {raw: gp.ArraySpec(voxel_size=voxel_size[:5], interpolatable=True)},
        )
        # + Print("A")
        + gp.Unsqueeze([raw], axis=0)
        # + Print("B")
        + gp.Normalize(raw, factor=predictconfig["normalization_factor"])
        # + Print("C")
#        + gp.Pad(raw, context, mode="reflect")
        + gp.Pad(raw, context[-5:])
        # + Print("D")
#        + gp.Pad(raw, context, mode="reflect")
        + predict
        # + Print("E")
        + gp.Squeeze([prediction],axis=0)
        # + Print("F")
        + gp.ZarrWrite(
            dataset_names={
                prediction: predictconfig["prediction_dataset_name"]
            },
            output_filename=predictconfig["prediction_container_path"],
            store=predictconfig["prediction_container_path"]
        )
        # + Print("G")
        + gp.Scan(scan_request)
    )

    request = gp.BatchRequest()
    # request to pipeline for ROI of whole image/volume
    with gp.build(pipeline):
        pipeline.request_batch(request)

    ds.attrs["axis_names"] = ["s", "c"] + ["t", "z", "y", "x"]


    ds.attrs["resolution"] = (1,) * predictconfig["num_spatial_dims"]
    ds.attrs["offset"] = (0,) * predictconfig["num_spatial_dims"]