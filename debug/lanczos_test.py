from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
import torch
x = torch.rand(1, 3, 128, 128)  # 4D tensor
resize(x, [64,64], interpolation=InterpolationMode.LANCZOS, antialias=True)