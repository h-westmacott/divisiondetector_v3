import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets_v2 import DivisonDatasetV2
from models import Unet4D
import torch
import torch.nn as nn
from utils import get_logger
from utils.losses import BCEDiceLoss, CEDiceLoss
import numpy as np
import zarr
# from utils.div_unet import DivUNet
# from funlib.learn.torch.models.conv4d import Conv4d


def __crop(tensor): # crop_factor is left/right padding for (z, y, x)
        t, d, h, w = 2, 2, 8 ,8
        return tensor[:, :, t:-t, d:-d, h:-h, w:-w].float()

def train_loop(dataloader, model, loss_fn, optimizer, logger, device, batch_size, save_snapshot_every=100, experiment_name =''):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    all_losses = []
    for batch, (X, y, a) in tqdm(enumerate(dataloader)):
        # Compute prediction and loss
        X = X.float().to(device)
        y = y.float().to(device)
        pred = model(X)
        loss = loss_fn(pred, __crop(y))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        all_losses.append(float(loss.detach().cpu().numpy()))
        logger.add(key="loss", value=float(loss.detach().cpu().numpy()))
        logger.write()
        logger.plot()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        if batch % save_snapshot_every == 0:
            save_snapshot(
                X,
                pred,
                __crop(y),
                batch,
                path = "/cephfs/henryjw/division_detector_data/results/",
                experiment_name = experiment_name
            )

    return all_losses

def save_model(state, iteration, is_lowest=False, experiment_name = ''):
    if experiment_name != '':
        model_path = os.path.join(experiment_name,"models")
        print('setting model path',experiment_name)
    else: 
        model_path = "models"

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if is_lowest:
        file_name = os.path.join(model_path, "best_loss.pth")
        torch.save(state, file_name)
        print(f"Best model weights saved at iteration {iteration}")
    
    file_name = os.path.join(model_path, str(iteration).zfill(6) + ".pth")
    torch.save(state, file_name)
    print(f"Checkpoint saved at iteration {iteration}")

def save_snapshot(batch, prediction, target, iteration, path = '', experiment_name = ''):
    if experiment_name != '':
        snapshot_path = os.path.join(experiment_name,"snapshots.zarr")
        print('setting snapshot path',experiment_name)
    else:
        snapshot_path = "snapshots.zarr"

    raw = batch
    # raw = raw[0,:]
    num_spatial_dims = len(raw.shape) - 2

    axis_names = ["s", "c"] + ["t", "z", "y", "x"][-num_spatial_dims:]
    prediction_offset = tuple(
        (a - b) / 2
        for a, b in zip(
            raw.shape[-num_spatial_dims:], prediction.shape[-num_spatial_dims:]
        )
    )
    f = zarr.open(path+'/'+snapshot_path, "a")
    f[f"{iteration}/raw"] = raw.detach().cpu().numpy()
    f[f"{iteration}/raw"].attrs["axis_names"] = axis_names
    f[f"{iteration}/raw"].attrs["resolution"] = [
        1,
    ] * num_spatial_dims

    # normalize the offsets by subtracting the mean offset per image
    prediction_cpu = prediction.detach().cpu().numpy()
    # prediction_cpu_reshaped = np.reshape(
    #     prediction_cpu, (prediction_cpu.shape[0], prediction_cpu.shape[1], -1)
    # )
    # mean_prediction = np.mean(prediction_cpu_reshaped, 2)
    # prediction_cpu -= mean_prediction[(...,) + (np.newaxis,) * (num_spatial_dims+1)]
    f[f"{iteration}/prediction"] = prediction_cpu
    f[f"{iteration}/prediction"].attrs["axis_names"] = axis_names
    f[f"{iteration}/prediction"].attrs["offset"] = prediction_offset
    f[f"{iteration}/prediction"].attrs["resolution"] = [
        1,
    ] * num_spatial_dims
    f[f"{iteration}/target"] = target.detach().cpu().numpy()
    f[f"{iteration}/target"].attrs["axis_names"] = axis_names
    f[f"{iteration}/target"].attrs["offset"] = prediction_offset
    f[f"{iteration}/target"].attrs["resolution"] = [
        1,
    ] * num_spatial_dims

def train(trainconfig):
    experiment_name = trainconfig["experiment_name"]
    learning_rate = trainconfig["learning_rate"]
    batch_size = trainconfig["batch_size"]
    epochs = trainconfig["epochs"]

    if experiment_name != '':
        if not os.path.exists(experiment_name):
            os.makedirs(experiment_name)

    # path_to_divisions = "X:/Exchange/Steffen/DivisionDetector/prediction_setup34_300000_180420_0.3.csv"
    path_to_divisions = trainconfig["path_to_divisions"]
    # img_dir = "X:/Guest/nho/sample_data/volumes.zarr"
    img_dir = trainconfig["img_dir"]
    dataset_name = trainconfig["dataset_name"]
    # training_data = DivisonDatasetV2(path_to_divisions, img_dir, window_size = (10,10,10),time_window_size=2)
    training_data = DivisonDatasetV2(path_to_divisions, img_dir, dataset_name, window_size = {'X':50,'Y':50,'Z':10},time_window_size=6)
    # train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Unet4D(in_channels=1,out_channels=1)
    model.train()
    model= nn.DataParallel(model)
    # device = torch.device('cuda:1')
    model = model.to(device)

    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    # loss_fn = CEDiceLoss(ce_weight=1.0, dice_weight=0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
    logger = get_logger(keys=["loss",
                                ], title=os.path.join(experiment_name,"loss"))

    lowest_loss = 1e6
    all_losses = []
    start_iteration = 0

    checkpoint = trainconfig["checkpoint"]

    if checkpoint == '':
        pass
    else:
        print(f"Resuming model from {checkpoint}")
        state = torch.load(checkpoint, map_location=device)
        start_iteration = state["iteration"] + 1
        lowest_loss = state["lowest_loss"]
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optim_state_dict"])
        logger.data = state["logger_data"]

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        loss = train_loop(train_dataloader, model, loss_fn, optimizer, logger, device, batch_size, experiment_name = experiment_name)
        print(scheduler.get_lr())
        scheduler.step()
        print(scheduler.get_lr())
        all_losses+=loss
        # test_loop(test_dataloader, model, loss_fn)
        is_lowest = np.mean(loss) < lowest_loss
        lowest_loss = min(lowest_loss, np.mean(loss))
        iteration = start_iteration + ((t+1)*len(train_dataloader))
        state = {
                "iteration": iteration,
                "lowest_loss": lowest_loss,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "logger_data": logger.data,
            }
        save_model(state, iteration, is_lowest, experiment_name)
    print("Done!")