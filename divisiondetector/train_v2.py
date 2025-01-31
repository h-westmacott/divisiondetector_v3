from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets_v2 import DivisonDatasetV2
from models import Unet4D
import torch
import torch.nn as nn
from utils import get_logger
# from utils.div_unet import DivUNet
# from funlib.learn.torch.models.conv4d import Conv4d


def __crop(tensor): # crop_factor is left/right padding for (z, y, x)
        t, d, h, w = 2, 2, 8 ,8
        return tensor[:, :, t:-t, d:-d, h:-h, w:-w].float()

def train_loop(dataloader, model, loss_fn, optimizer, logger):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    model.to('cuda:0')
    all_losses = []
    for batch, (X, y, a) in tqdm(enumerate(dataloader)):
        # Compute prediction and loss
        X = X.float().to('cuda:0')
        y = y.float().to('cuda:0')
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


learning_rate = 1e-3
batch_size = 4
epochs = 1

# path_to_divisions = "X:/Exchange/Steffen/DivisionDetector/prediction_setup34_300000_180420_0.3.csv"
path_to_divisions = r"X:\Guest\nho\data\processed_annotations\positive_divisions_160616.csv"
# img_dir = "X:/Guest/nho/sample_data/volumes.zarr"
img_dir = r"X:\Guest\nho\data\160616.zarr"
dataset_name = 'raw'
# training_data = DivisonDatasetV2(path_to_divisions, img_dir, window_size = (10,10,10),time_window_size=2)
training_data = DivisonDatasetV2(path_to_divisions, img_dir, dataset_name, window_size = {'X':50,'Y':50,'Z':10},time_window_size=6)
# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

model = Unet4D(in_channels=1,out_channels=1)
model = model.to('cuda:0')

loss_fn = nn.BCEWithLogitsLoss()
# loss_fn = BCEDiceLoss(bce_weight=0.7, dice_weight=0.3)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

logger = get_logger(keys=["loss",
                            ], title="loss")

all_losses = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss = train_loop(train_dataloader, model, loss_fn, optimizer, logger)
    all_losses+=loss
    # test_loop(test_dataloader, model, loss_fn)
print("Done!")