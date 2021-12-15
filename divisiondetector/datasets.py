from torch.utils.data.dataset import Dataset
import numpy as np

class DivisionDataset(Dataset):

    def __init__(self):

        super().__init__()
        # TODO: initialize gunpowder pipeline here

    def __len__(self):
        if not hasattr(self, "_length"):
            # TODO: compute length of the dataset here
            self._length = 20
        return self._length

    def __getitem__(self, idx):

        # TODO: call request for pipeline here
        # Anything below is just a placeholder
        channels = 2
        time_window = 3
        depth = 8
        width = 128
        height = 128
        
        network_input = np.random.rand(channels,
                                       time_window,
                                       depth,
                                       width,
                                       height)
        
        target = np.random.rand(1,
                                time_window,
                                depth,
                                width,
                                height)
        
        network_input = network_input.astype(np.float32)
        target = target.astype(np.float32)
        
        return network_input, target
