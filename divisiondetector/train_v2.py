import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets_v2 import DivisonDatasetV2
from models import Unet4D
import torch
import torch.nn as nn
from utils import get_logger
import numpy as np
# from utils.div_unet import DivUNet
# from funlib.learn.torch.models.conv4d import Conv4d


def __crop(tensor): # crop_factor is left/right padding for (z, y, x)
        t, d, h, w = 2, 2, 8 ,8
        return tensor[:, :, t:-t, d:-d, h:-h, w:-w].float()

def train_loop(dataloader, model, loss_fn, optimizer, logger,device):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    # model.train()
    # model.to(device)
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
    return all_losses

def save_model(state, iteration, is_lowest=False):
    if not os.path.exists("models"):
        os.makedirs("models")
    if is_lowest:
        file_name = os.path.join("models", "best_loss.pth")
        torch.save(state, file_name)
        print(f"Best model weights saved at iteration {iteration}")
    
    file_name = os.path.join("models", str(iteration).zfill(6) + ".pth")
    torch.save(state, file_name)
    print(f"Checkpoint saved at iteration {iteration}")

learning_rate = 1e-3
batch_size = 64
epochs = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# path_to_divisions = "X:/Exchange/Steffen/DivisionDetector/prediction_setup34_300000_180420_0.3.csv"
path_to_divisions = r"/cephfs/henryjw/division_detector_data/positive_divisions_160616.csv"
# img_dir = "X:/Guest/nho/sample_data/volumes.zarr"
img_dir = r"/cephfs/henryjw/division_detector_data/160616.zarr"
dataset_name = 'raw'
# training_data = DivisonDatasetV2(path_to_divisions, img_dir, window_size = (10,10,10),time_window_size=2)
training_data = DivisonDatasetV2(path_to_divisions, img_dir, dataset_name, window_size = {'X':50,'Y':50,'Z':10},time_window_size=6)
# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=64)

model = Unet4D(in_channels=1,out_channels=1)
model.train()
model= nn.DataParallel(model)
model = model.to(device)

loss_fn = nn.BCEWithLogitsLoss()
# loss_fn = BCEDiceLoss(bce_weight=0.7, dice_weight=0.3)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

logger = get_logger(keys=["loss",
                            ], title="loss")
lowest_loss = 1e6
all_losses = []
start_iteration = 0

checkpoint = 'models/best_loss.pth'

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
    loss = train_loop(train_dataloader, model, loss_fn, optimizer, logger, device)
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
    save_model(state, iteration, is_lowest)

print("Done!")