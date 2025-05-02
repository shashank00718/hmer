import torch
from torchvision.models import densenet121, DenseNet121_Weights

# Load the model with the recommended way
weights = DenseNet121_Weights.IMAGENET1K_V1  # or DenseNet121_Weights.DEFAULT
model = densenet121(weights=weights)

# Save the model's state_dict
torch.save(model.state_dict(), "densenet121_pretrained.pth")
