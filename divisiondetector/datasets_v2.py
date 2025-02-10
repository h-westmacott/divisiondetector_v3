# import os
import random
import pandas as pd
import gunpowder as gp
from torch.utils.data import Dataset
import zarr
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from torchvision.transforms import v2
# from torchvision.io import read_image

class DivisonDatasetV2(Dataset):
    def __init__(self, path_to_divisions, img_dir, dataset_name, window_size, time_window_size, transform=None, target_transform=None, num_channels=1):
        self.divisions_pd = pd.read_csv(path_to_divisions, header=None)
        self.divisions_pd.columns = ["T", "Z", "Y", "X", "ID"]
        self.divisions_pd = self.divisions_pd.reset_index()
        self.img_dir = img_dir
        self.dataset_name = dataset_name
        self.window_size = window_size
        self.time_window_size = time_window_size
        self.zarr_dataset = zarr.open(img_dir+'//'+self.dataset_name)
        self.target_transform = target_transform
        self.num_dims = len(self.zarr_dataset.shape)
        self.num_channels = self.zarr_dataset.shape[0]
        self.dataset_size = {"Z" : self.zarr_dataset.shape[2],
                             "Y" : self.zarr_dataset.shape[3],
                             "X" : self.zarr_dataset.shape[4],
                             "T" : self.zarr_dataset.shape[1]}
                                                          
        self.offset_augment = True
        self.normalization_factor = 1/512
        self.__setup_pipeline()

    def __len__(self):
        return len(self.divisions_pd)*25

    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        # return image, label
        
        
        # if idx<len(self.divisions_pd):
        if random.random()>0.4:
            # True division
            idx = idx%len(self.divisions_pd)
            this_detection = self.divisions_pd.astype('int64').iloc[idx]
            this_detection["Z"] = this_detection["Z"]/5
            # vol = self.pipeline.fetch(
            # (this_detection["Z"], this_detection["Y"], this_detection["X"],this_detection['T']),
            # (self.window_size,self.time_window_size)
            # )
            true_division = True
            if self.offset_augment:
                # We want to add a small postive or negative value to X, Y, Z and/or T
                # We need this offset to be recorded, so that the same offset can be removed(?) from the ground truth dataset.
                # Sigma should be related to window size
                # offset_t = int(np.floor(random.gauss(mu=0.0, sigma=3.0)))
                offset_t = 0
                offset_z = int(np.floor(random.gauss(mu=0.0, sigma=3.0)))
                offset_y = int(np.floor(random.gauss(mu=0.0, sigma=3.0)))
                offset_x = int(np.floor(random.gauss(mu=0.0, sigma=3.0)))

                this_detection["T"] = this_detection["T"] + offset_t
                this_detection["Z"] = this_detection["Z"] + offset_z
                this_detection["Y"] = this_detection["Y"] + offset_y
                this_detection["X"] = this_detection["X"] + offset_x

            request = gp.BatchRequest()
            request[self.raw] = gp.ArraySpec(
                roi=gp.Roi(
                    (0,
                    min((self.dataset_size["T"]-1)-(self.time_window_size),max(0, this_detection["T"]-(self.time_window_size//2))),
                    min(((self.dataset_size["Z"])-1)-(self.window_size['Z']),max(0, this_detection["Z"]-(self.window_size['Z']//2))),
                    min((self.dataset_size["Y"]-1)-(self.window_size['Y']),max(0, this_detection["Y"]-(self.window_size['Y']//2))),
                    min((self.dataset_size["X"]-1)-(self.window_size['X']),max(0, this_detection["X"]-(self.window_size['X']//2))),),
                    # (1, self.num_channels, *self.crop_size),
                    (self.num_channels, self.time_window_size, self.window_size['Z'],self.window_size['Y'],self.window_size['X']),
                )
            )
            with gp.build(self.pipeline):
                vol = self.pipeline.request_batch(request)
            raw = vol[self.raw].data
        else:
            raw_max = 0.0
            while raw_max == 0.0:
                this_detection = {"Z" : random.randint(0,((self.dataset_size["Z"])-1)-(self.window_size['Z'])),
                                    "Y" : random.randint(0,(self.dataset_size["Y"]-1)-(self.window_size['Y'])),
                                    "X" : random.randint(0,(self.dataset_size["X"]-1)-(self.window_size['X'])),
                                    "T" : random.randint(0,(self.dataset_size["T"]-1)-(self.time_window_size))}

                # vol = self.pipeline.fetch(
                # (this_detection["Z"], this_detection["Y"], this_detection["X"],this_detection['T']),
                # (self.window_size,self.time_window_size)
                # )
                true_division = False
            
                request = gp.BatchRequest()
                request[self.raw] = gp.ArraySpec(
                    roi=gp.Roi(
                        (0,
                        min((self.dataset_size["T"]-1)-(self.time_window_size),max(0, this_detection["T"]-(self.time_window_size//2))),
                        min(((self.dataset_size["Z"])-1)-(self.window_size['Z']),max(0, this_detection["Z"]-(self.window_size['Z']//2))),
                        min((self.dataset_size["Y"]-1)-(self.window_size['Y']),max(0, this_detection["Y"]-(self.window_size['Y']//2))),
                        min((self.dataset_size["X"]-1)-(self.window_size['X']),max(0, this_detection["X"]-(self.window_size['X']//2))),),
                        # (1, self.num_channels, *self.crop_size),
                        (self.num_channels, self.time_window_size, self.window_size['Z'],self.window_size['Y'],self.window_size['X']),
                    )
                )
                with gp.build(self.pipeline):
                    vol = self.pipeline.request_batch(request)
                raw = vol[self.raw].data
                raw_max = raw.max()

                # check if bounding box around this_detection contains any of the points in self.divisions_pd:

            #     print(raw.max())
            # print('max of raw being taken forward:',raw.max())
        ground_truth = np.zeros_like(vol[self.raw].data)
        if true_division:
            # We do have a real division, with coordinates defined by this_detection.
            # What we need to know is where this coordinate lies relative to the rest of the requested ROI.
            
            t_start = min((self.dataset_size["T"] - 1) - self.time_window_size, max(0, this_detection["T"] - (self.time_window_size // 2)))
            z_start = min((self.dataset_size["Z"] - 1) - self.window_size['Z'], max(0, this_detection["Z"] - (self.window_size['Z'] // 2)))
            y_start = min((self.dataset_size["Y"] - 1) - self.window_size['Y'], max(0, this_detection["Y"] - (self.window_size['Y'] // 2)))
            x_start = min((self.dataset_size["X"] - 1) - self.window_size['X'], max(0, this_detection["X"] - (self.window_size['X'] // 2)))
            
            # Compute the relative coordinates within the bounding box
            relative_t = this_detection["T"] - t_start
            relative_z = this_detection["Z"] - z_start
            relative_y = this_detection["Y"] - y_start
            relative_x = this_detection["X"] - x_start

            if self.offset_augment:
                relative_t -= offset_t
                relative_z -= offset_z
                relative_y -= offset_y
                relative_x -= offset_x

            # if any of these relative values are now negative, we gonna have a problem.
            # We need to pad the array with zeros, up to that negative value. We then replace 
            # that negative value with a zero, and crop in afterwards. 
            # Similarly, there's probably a case where the relative value lies outside the bounds of the image. 
            # if any([relative_t, relative_z, relative_y, relative_x])<0:
            
            # The simpler, abut less efficient solution is to pad the array by a set amount in all dimensions, 
            # and justa dd that pad value to relative_t, relative_z ...., then crop it in again afterwards. 
            pad_width = 10
            ground_truth = np.pad(ground_truth,((0,0),(pad_width,pad_width),(pad_width,pad_width),(pad_width,pad_width),(pad_width,pad_width)))

            ground_truth[0,relative_t+pad_width, relative_z+pad_width, relative_y+pad_width, relative_x+pad_width] = 1

            ground_truth = apply_4d_gaussian_blur(ground_truth, sigma=(0,1,8/5,8,8))

            ground_truth = ground_truth[:, pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width]


        # return vol[self.raw].data, ground_truth, true_division, dict(this_detection)
        return raw, ground_truth, true_division

    def __setup_pipeline(self):
        self.raw = gp.ArrayKey("RAW")

        # treat all dimensions as spatial, with a voxel size of 1
        raw_spec = gp.ArraySpec(voxel_size=(1,) * self.num_dims, interpolatable=True)

        # spatial_dims = tuple(range(self.num_dims - self.num_spatial_dims,
        # self.num_dims))

        self.pipeline = (
            gp.ZarrSource(
                self.img_dir,
                {self.raw: 'raw'},
                array_specs={self.raw: raw_spec},
            )
            + gp.Normalize(self.raw, factor=self.normalization_factor)
        )
        
def apply_4d_gaussian_blur(ground_truth, sigma):
    """
    Applies a 4D Gaussian blur to a ground truth array.

    Parameters:
        ground_truth (ndarray): A 4D array with ground truth division points (1 at division locations, 0 elsewhere).
        sigma (tuple): A tuple of standard deviations (sigma_T, sigma_Z, sigma_Y, sigma_X) for the Gaussian kernel.

    Returns:
        ndarray: A 4D array with Gaussian blurred values.
    """
    # Ensure the input is a 4D array
    # assert ground_truth.ndim == 4, "Ground truth array must be 4D (T, Z, Y, X)"
    
    # Apply Gaussian filtering
    blurred_array = gaussian_filter(ground_truth, sigma=sigma)
    blurred_array = blurred_array/np.amax(blurred_array)
    return blurred_array