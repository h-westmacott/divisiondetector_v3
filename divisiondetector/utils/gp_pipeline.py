import gunpowder as gp
from gunpowder.nodes import simple_augment
import math
import zarr

class GPPipeline():
    def __init__(self, vol_path, div_path, mode, ball_radius): # Ball radius (z, y, x)
        def compile_pipeline(vol_key, div_key, divr_key):
            # EDIT GUNPOWDER PIPELINE HERE!
            vol_source = (gp.ZarrSource(
                self.vol_path,
                {vol_key: 'raw'},
                {vol_key: gp.ArraySpec( # Resolution used as voxel size
                    interpolatable=True
                )}
            ) + gp.Pad(vol_key, None))

            div_source = (gp.CsvPointsSource(
                self.div_path,
                div_key,
                ndims=4, # (Timepoint, Z, Y, X)
            ) + gp.Pad(div_key, None))

            merge_prov = gp.MergeProvider()

            rasterise_divs = gp.RasterizeGraph(
                div_key,
                divr_key,
                settings=gp.RasterizationSettings(
                    radius=(0.1, ball_radius[0], ball_radius[1], ball_radius[2]), # Time radius < 1 to tell gunpowder that there is no radius (only return points from that Timepoint)
                    mode=mode
                )
            )

            normalise = gp.Normalize(vol_key)
            simple_augment = gp.SimpleAugment(mirror_only=[1, 2, 3], transpose_only=[1, 2, 3])

            pipeline = (
                (div_source, vol_source) +
                merge_prov +
                normalise + 
                #simple_augment +
                rasterise_divs
            )

            return pipeline

        def get_resolution(source_path):
            f = zarr.open(source_path, 'r')
            return f['raw'].attrs['resolution']

        self.vol_path = vol_path
        self.div_path = div_path

        self.resolution = get_resolution(self.vol_path) # Used as voxel size

        self.vol_key = gp.ArrayKey('RAW') # Key for volume
        self.div_key = gp.GraphKey('DIV') # Key for divisions
        self.divr_key = gp.ArrayKey('DIVRAS') # Key for rasterised divisions

        # Check rasterisation mode. If invalid, default to 'ball'
        if mode != 'ball' and mode != 'peak':
            mode = 'ball'
            print("Rasterisation mode not valid. Defaulting to 'ball'.")

        self.pipeline = compile_pipeline(
            self.vol_key,
            self.div_key,
            self.divr_key
        )

        self.pipeline.setup()

    def fetch_data(self, coords, window_size, timepoint, time_window): # Coords, window size (z, y, x); Time window (pre, post)

        def round_to_res_window(i, coords, res):
            return res[i+1] * (int(coords[i] - math.ceil(window_size[i]/2)) // res[i+1])
        
        request = gp.BatchRequest()
        
        request[self.vol_key] = gp.Roi(
            (
                int(timepoint - time_window[0]),
                round_to_res_window(0, coords, self.resolution),
                round_to_res_window(1, coords, self.resolution),
                round_to_res_window(2, coords, self.resolution)
            ),
            (sum(list(time_window))+1, window_size[0], window_size[1], window_size[2])
        )

        label_roi = gp.Roi(
            (
                timepoint,
                int(coords[0] - math.ceil(window_size[0]/2)),
                int(coords[1] - math.ceil(window_size[1]/2)),
                int(coords[2] - math.ceil(window_size[2]/2))
            ),
            (1, window_size[0], window_size[1], window_size[2])
        )
        
        request[self.divr_key] = label_roi

        request[self.div_key] = label_roi

        batch = self.pipeline.request_batch(request)

        return batch[self.vol_key].data, batch[self.divr_key].data, batch[self.div_key]

    def clear(self):
        self.pipeline.internal_teardown()