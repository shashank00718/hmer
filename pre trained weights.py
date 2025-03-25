import torch
import torchvision.models as models

model = models.densenet121(pretrained=True)  # Automatically downloads weights
