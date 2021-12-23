import torch
import torch.nn as nn
from .utils.div_unet import DivUNet
from funlib.learn.torch.models.conv4d import Conv4d

class Unet4D(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            num_fmaps=64,
            fmap_inc_factor=3,
            features_in_last_layer=64,
            depth=1):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features_in_last_layer = features_in_last_layer

        d_factors = [(1, 2, 2), ] * depth
        self.backbone = DivUNet(in_channels=self.in_channels,
                             num_fmaps=num_fmaps,
                             fmap_inc_factor=fmap_inc_factor,
                             downsample_factors=d_factors,
                             activation='ReLU',
                             padding='valid',
                             num_fmaps_out=self.features_in_last_layer,
                            # TODO: check if this needs to be changed to (1,1,3,3)
                             kernel_size_down=[[(1, 1, 3, 3), (1,1,1,1), (3, 3, 3, 3)]] * (depth + 1),
                             kernel_size_up=[[(1, 1, 3, 3), (1, 1, 3, 3)]] * depth, # (1, 1, 3, 3), (1, THIS ONE MISBEHAVES -> 1, 3, 3)
                             constant_upsample=True)

        self.head = torch.nn.Sequential(Conv4d(self.features_in_last_layer, self.features_in_last_layer, (1, 1, 1, 1)),
                                        nn.ReLU(),
                                        Conv4d(self.features_in_last_layer, out_channels, (1, 1, 1, 1)))

    def forward(self, raw):
        h = self.backbone(raw)
        return self.head(h)

