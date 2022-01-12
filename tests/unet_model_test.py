from divisiondetector.models import Unet4D
import torch

unet = Unet4D(1, 1, 16, depth=2).cuda()
input_data = torch.rand(1, 1, 7, 32, 132, 132).cuda() # z-axis already divided by 5; for a fresh dataset, dimension should be 160

output = unet(input_data)
# print(output)
print(output.shape)