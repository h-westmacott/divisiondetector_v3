import torch
import torch.nn as nn
from utils.div_unet import DivUNet
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
                            #  padding='valid',
                             padding='same',
                             num_fmaps_out=self.features_in_last_layer,
                             kernel_size_down=[[(1, 1, 3, 3), (1,1,1,1), (3, 3, 3, 3)]] * (depth + 1),
                             kernel_size_up=[[(1, 1, 3, 3), (1, 1, 3, 3)]] * depth, # (1, 1, 3, 3), (1, This one alters other dimensions -> 1, 3, 3)
                             constant_upsample=True)

        self.head = torch.nn.Sequential(Conv4d(self.features_in_last_layer, self.features_in_last_layer, (1, 1, 1, 1)),
                                        nn.ReLU(),
                                        Conv4d(self.features_in_last_layer, self.features_in_last_layer, (1, 1, 1, 1)),
                                        nn.ReLU(),
                                        Conv4d(self.features_in_last_layer, out_channels, (1, 1, 1, 1)),
                                        # nn.Sigmoid(),
                                        )

    def forward(self, raw):
        # if raw.max()<=0:
        #     return torch.zeros((1,1,2,6,34,34),device=raw.device)
        # print(raw.shape)
        h = self.backbone(raw)
        return self.head(h)

# import torch
# import torch.nn as nn
import torch.nn.functional as F

class DivisionDetector3D(nn.Module):
    def __init__(self, model_spec):
        super(DivisionDetector3D, self).__init__()
        self.model_spec = model_spec
        self.n_conv_filters = model_spec['n_conv_filters']
        self.n_output_hus = model_spec['n_output_hus']
        self.activation = self._get_activation(model_spec['activation'])
        self.batch_norm = model_spec['batch_norm']
        self.output_bn = model_spec['output_bn']
        self.residual = model_spec['residual']

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        xy_fws = [13, 11, 7, 9, 3, 3, 3, 3]
        z_fws = [1, 1, 1, 1, 3, 3, 3, 3]

        for z_fw, xy_fw, n_filter in zip(z_fws, xy_fws, self.n_conv_filters):
            self.conv_layers.append(nn.Conv3d(in_channels=7 if len(self.conv_layers) == 0 else self.n_conv_filters[len(self.conv_layers) - 1],
                                              out_channels=n_filter,
                                              kernel_size=(z_fw, xy_fw, xy_fw),
                                              padding=(0, xy_fw // 2, xy_fw // 2),
                                              bias=not self.batch_norm))
            if self.batch_norm:
                self.bn_layers.append(nn.BatchNorm3d(n_filter))

        self.output_layers = nn.ModuleList()
        self.output_bn_layers = nn.ModuleList()

        for n_hu in self.n_output_hus + [1]:
            self.output_layers.append(nn.Conv3d(in_channels=self.n_conv_filters[-1] if len(self.output_layers) == 0 else self.n_output_hus[len(self.output_layers) - 1],
                                                out_channels=n_hu,
                                                kernel_size=(1, 1, 1),
                                                bias=not self.output_bn))
            if self.output_bn:
                self.output_bn_layers.append(nn.BatchNorm3d(n_hu))

    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'LeakyReLU':
            return nn.LeakyReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        residuals = []
        x = x.squeeze(1)
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if self.batch_norm:
                x = self.bn_layers[i](x)
            x = self.activation(x)
            if self.residual and i > 0 and x.shape == residuals[-1].shape:
                x = x + residuals.pop()
            residuals.append(x)

        for i, output_layer in enumerate(self.output_layers):
            x = output_layer(x)
            if self.output_bn:
                x = self.output_bn_layers[i](x)
            if i < len(self.output_layers) - 1:
                x = self.activation(x)
            else:
                x = torch.sigmoid(x)
        x = x.unsqueeze(1)
        return x
    


class DivisionDetector4D(nn.Module):
    def __init__(self, model_spec):
        super(DivisionDetector4D, self).__init__()
        self.model_spec = model_spec
        self.n_conv_filters = model_spec['n_conv_filters']
        self.n_output_hus = model_spec['n_output_hus']
        self.activation = self._get_activation(model_spec['activation'])
        self.batch_norm = model_spec['batch_norm']
        self.output_bn = model_spec['output_bn']
        self.residual = model_spec['residual']

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        xy_fws = [13, 11, 7, 9, 3, 3, 3, 3]
        z_fws = [1, 1, 1, 1, 3, 3, 3, 3]

        for z_fw, xy_fw, n_filter in zip(z_fws, xy_fws, self.n_conv_filters):
            self.conv_layers.append(Conv4d(in_channels=7 if len(self.conv_layers) == 0 else self.n_conv_filters[len(self.conv_layers) - 1],
                                              out_channels=n_filter,
                                              kernel_size=(z_fw, xy_fw, xy_fw),
                                              padding=(0, xy_fw // 2, xy_fw // 2),
                                              bias=not self.batch_norm))
            if self.batch_norm:
                self.bn_layers.append(nn.BatchNorm3d(n_filter))

        self.output_layers = nn.ModuleList()
        self.output_bn_layers = nn.ModuleList()

        for n_hu in self.n_output_hus + [1]:
            self.output_layers.append(Conv4d(in_channels=self.n_conv_filters[-1] if len(self.output_layers) == 0 else self.n_output_hus[len(self.output_layers) - 1],
                                                out_channels=n_hu,
                                                kernel_size=(1, 1, 1),
                                                bias=not self.output_bn))
            if self.output_bn:
                self.output_bn_layers.append(nn.BatchNorm3d(n_hu))

    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'LeakyReLU':
            return nn.LeakyReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        residuals = []
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if self.batch_norm:
                x = self.bn_layers[i](x)
            x = self.activation(x)
            if self.residual and i > 0 and x.shape == residuals[-1].shape:
                x = x + residuals.pop()
            residuals.append(x)

        for i, output_layer in enumerate(self.output_layers):
            x = output_layer(x)
            if self.output_bn:
                x = self.output_bn_layers[i](x)
            if i < len(self.output_layers) - 1:
                x = self.activation(x)
            else:
                x = torch.sigmoid(x)

        return x