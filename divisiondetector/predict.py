import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import Unet4D, DivisionDetector3D
import torch
import torch.nn as nn
import numpy as np
import gunpowder as gp
import zarr
from datasets_prediction import PredictionDivisonDatasetV2
# from utils.div_unet import DivUNet
# from funlib.learn.torch.models.conv4d import Conv4d

def predict(predictconfig):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Unet4D(in_channels=1,out_channels=1)
    model_spec = {
        'n_conv_filters': [32, 64, 128, 256, 512, 512, 512, 512],
        'n_output_hus': [64, 64],
        'activation': 'relu',
        'batch_norm': True,
        'output_bn': True,
        'residual': True,
        'train_spec': {
            'learning_rate': 0.001,
            'clip_grads': True,
            'partial_weight': 1e-4
        }
    }
    model = DivisionDetector3D(model_spec)
    
    model= nn.DataParallel(model)
    device = torch.device('cuda:0')
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
        Print("starting")
        + gp.ZarrSource(
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
        # + Print("D: stacking")
        # + gp.Stack(num_repetitions=64)
        # + Print("E")
#        + gp.Pad(raw, context, mode="reflect")
        + predict
        # + Print("F")
        + gp.Squeeze([prediction],axis=0)
        # + Print("G")
        + gp.ZarrWrite(
            dataset_names={
                prediction: predictconfig["prediction_dataset_name"]
            },
            output_filename=predictconfig["prediction_container_path"],
            store=predictconfig["prediction_container_path"]
        )
        # + Print("H")
        + gp.Scan(scan_request)
    )

    request = gp.BatchRequest()
    # request to pipeline for ROI of whole image/volume
    with gp.build(pipeline):
        pipeline.request_batch(request)

    ds.attrs["axis_names"] = ["s", "c"] + ["t", "z", "y", "x"]


    ds.attrs["resolution"] = (1,) * predictconfig["num_spatial_dims"]
    ds.attrs["offset"] = (0,) * predictconfig["num_spatial_dims"]


def predict_v2(predictconfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Unet4D(in_channels=1,out_channels=1)
    model_spec = {
        'n_conv_filters': [32, 64, 128, 256, 512, 512, 512, 512],
        'n_output_hus': [64, 64],
        'activation': 'relu',
        'batch_norm': True,
        'output_bn': True,
        'residual': True,
        'train_spec': {
            'learning_rate': 0.001,
            'clip_grads': True,
            'partial_weight': 1e-4
        }
    }
    model = DivisionDetector3D(model_spec)
    
    
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

    prediction_data = PredictionDivisonDatasetV2(img_dir = predictconfig["raw_container_path"], dataset_name=predictconfig["raw_dataset_name"], batch_size = predictconfig['batch_size'], window_size = {'X':50,'Y':50,'Z':10},time_window_size=7)
    prediction_dataloader = DataLoader(prediction_data, batch_size=predictconfig["batch_size"], shuffle=False, num_workers=0)

    # container = zarr.open(predictconfig["raw_container_path"])
    # prediction = container.create_array(
    #     shape=(1, prediction_data.dataset_size['T'], prediction_data.dataset_size['Z'], prediction_data.dataset_size['Y'], prediction_data.dataset_size['X']),
    #     chunks=(1, prediction_data.dataset_size['T'], prediction_data.window_size['Z'],prediction_data.window_size['Y'], prediction_data.window_size['X']),
    #     dtype=np.float32, mode='a'
    # )

    container = zarr.open(predictconfig["raw_container_path"])
    data = container[predictconfig["raw_dataset_name"]]
    spatial_array = data.shape[-4:]
    # prepare the zarr dataset to write to
    f = zarr.open(predictconfig["prediction_container_path"])
    prediction = f.create_dataset(
        predictconfig["prediction_dataset_name"],
        shape=(
            # 1,
            1,
            *spatial_array,
        ),
        dtype=float,
        overwrite=True
    )
    print(len(prediction_dataloader))
    for batch, (X, loc) in tqdm(enumerate(prediction_dataloader)):
        X = X.float().to(device)
        pred = model(X)
        # write to zarr
        for patch in range(pred.shape[0]):
            # print(loc[patch])
            # print(np.amax(pred[patch].detach().cpu().numpy()))
            print(loc[patch])
            prediction[:,(loc[patch][3,0]+context[1]):(loc[patch][3,1]-context[1]),(loc[patch][2,0]+context[2]):(loc[patch][2,1]-context[2]),(loc[patch][1,0]+context[3]):(loc[patch][1,1]-context[3]),(loc[patch][0,0]+context[4]):(loc[patch][0,1]-context[4])] = pred[patch].detach().cpu().numpy()