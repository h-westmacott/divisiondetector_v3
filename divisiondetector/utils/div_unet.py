from funlib.learn.torch.models import ConvPass
from funlib.learn.torch.models.unet import Downsample
from funlib.learn.torch.models.unet import Upsample
from funlib.learn.torch.models.unet import UNet
import math
import torch
import torch.nn as nn

# Unet from funlib.learn.torch, edited to take in 5D data

class DivDownsample(Downsample):

    def forward(self, x):

        is_4d = len(x.shape) == 6
        
        if is_4d:
            b, c, t, d, h, w = x.shape
            x_reshaped = x.view((b, c*t, d, h, w))
        else:
            x_reshaped = x

        for d in range(1, self.dims + 1):
            if x.size()[-d] % self.downsample_factor[-d] != 0:
                raise RuntimeError(
                    "Can not downsample shape %s with factor %s, mismatch "
                    "in spatial dimension %d" % (
                        x.size(),
                        self.downsample_factor,
                        self.dims - d))

        x_downsampled = self.down(x_reshaped)

        if is_4d:
            dd, dh, dw = x_downsampled.shape[-3:]
            x_downsampled = x_downsampled.view(b, c, t, dd, dh, dw)

        return x_downsampled


class DivUpsample(Upsample):

    def crop_to_factor(self, x, factor, kernel_sizes):
        '''Crop feature maps to ensure translation equivariance with stride of
        upsampling factor. This should be done right after upsampling, before
        application of the convolutions with the given kernel sizes.

        The crop could be done after the convolutions, but it is more efficient
        to do that before (feature maps will be smaller).
        '''

        shape = x.size()
        spatial_shape = shape[-self.dims:]

        # the crop that will already be done due to the convolutions
        convolution_crop = tuple(
            sum(ks[d] - 1 for ks in kernel_sizes)
            for d in range(self.dims)
        )

        ns = (
            int(math.floor(float(s - c)/f))
            for s, c, f in zip(spatial_shape, convolution_crop, factor)
        )
        target_spatial_shape = tuple(
            n*f + c
            for n, c, f in zip(ns, convolution_crop, factor)
        )

        if target_spatial_shape != spatial_shape:

            assert all((
                    (t > c) for t, c in zip(
                        target_spatial_shape,
                        convolution_crop))
                ), \
                "Feature map with shape %s is too small to ensure " \
                "translation equivariance with factor %s and following " \
                "convolutions %s" % (
                    shape,
                    factor,
                    kernel_sizes)

            return self.crop(x, target_spatial_shape)

        return x

    def crop(self, x, shape):
        '''Center-crop x to match spatial dimensions given by shape.'''

        x_target_size = x.size()[:-len(shape)] + shape

        offset = tuple(
            (a - b)//2
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, f_left, g_out):

        is_4d = len(g_out.shape) == 6
        
        if is_4d:
            b, c, t, d, h, w = g_out.shape
            g_reshaped = g_out.view((b, c*t, d, h, w))
        else:
            g_reshaped = g_out

        g_up = self.up(g_reshaped)

        if self.next_conv_kernel_sizes is not None:
            g_cropped = self.crop_to_factor(
                g_up,
                self.crop_factor,
                self.next_conv_kernel_sizes)
        else:
            g_cropped = g_up

        if is_4d:
            dd,dh,dw = g_cropped.shape[-3:]
            g_cropped = g_cropped.view(b, c, t, dd, dh, dw)

        if is_4d:
            f_cropped = self.crop(f_left, g_cropped.size()[-self.dims-1:])
        else:
            f_cropped = self.crop(f_left, g_cropped.size()[-self.dims:])

        return torch.cat([f_cropped, g_cropped], dim=1)


class DivUNet(UNet):

    def __init__(
            self,
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down=None,
            kernel_size_up=None,
            activation='ReLU',
            fov=(1, 1, 1),
            voxel_size=(1, 1, 1),
            num_fmaps_out=None,
            num_heads=1,
            constant_upsample=False,
            padding='valid'):

        super().__init__(
            in_channels,
            num_fmaps,
            fmap_inc_factor,
            downsample_factors,
            kernel_size_down,
            kernel_size_up,
            activation,
            fov,
            voxel_size,
            num_fmaps_out,
            num_heads,
            constant_upsample,
            padding
        )

        self.num_levels = len(downsample_factors) + 1
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.out_channels = num_fmaps_out if num_fmaps_out else num_fmaps

        # default arguments

        if kernel_size_down is None:
            kernel_size_down = [[(3, 3, 3), (3, 3, 3)]]*self.num_levels
        if kernel_size_up is None:
            kernel_size_up = [[(3, 3, 3), (3, 3, 3)]]*(self.num_levels - 1)

        # compute crop factors for translation equivariance
        crop_factors = []
        factor_product = None
        for factor in downsample_factors[::-1]:
            if factor_product is None:
                factor_product = list(factor)
            else:
                factor_product = list(
                    f*ff
                    for f, ff in zip(factor, factor_product))
            crop_factors.append(factor_product)
        crop_factors = crop_factors[::-1]

        # modules

        # left convolutional passes
        self.l_conv = nn.ModuleList([
            ConvPass(
                in_channels
                if level == 0
                else num_fmaps*fmap_inc_factor**(level - 1),
                num_fmaps*fmap_inc_factor**level,
                kernel_size_down[level],
                activation=activation,
                padding=padding)
            for level in range(self.num_levels)
        ])
        self.dims = self.l_conv[0].dims

        # left downsample layers
        self.l_down = nn.ModuleList([
            DivDownsample(downsample_factors[level])
            for level in range(self.num_levels - 1)
        ])

        # right up/crop/concatenate layers
        self.r_up = nn.ModuleList([
            nn.ModuleList([
                DivUpsample(
                    downsample_factors[level],
                    mode='nearest' if constant_upsample else 'transposed_conv',
                    in_channels=num_fmaps*fmap_inc_factor**(level + 1),
                    out_channels=num_fmaps*fmap_inc_factor**(level + 1),
                    crop_factor=crop_factors[level],
                    next_conv_kernel_sizes=kernel_size_up[level])
                for level in range(self.num_levels - 1)
            ])
            for _ in range(num_heads)
        ])

        # right convolutional passes
        self.r_conv = nn.ModuleList([
            nn.ModuleList([
                ConvPass(
                    num_fmaps*fmap_inc_factor**level +
                    num_fmaps*fmap_inc_factor**(level + 1),
                    num_fmaps*fmap_inc_factor**level
                    if num_fmaps_out is None or level != 0
                    else num_fmaps_out,
                    kernel_size_up[level],
                    activation=activation,
                    padding=padding)
                for level in range(self.num_levels - 1)
            ])
            for _ in range(num_heads)
        ])