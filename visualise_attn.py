import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

image_path = r"C:\Users\shash\Downloads\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\off_image_train\off_image_train"
caption_file_path = r"C:\Users\shash\Downloads\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\train_caption.txt"

captions = {}
with open(caption_file_path, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        image_filename = parts[0]
        latex_expr = parts[1]
        captions[image_filename] = latex_expr

transform = transforms.Compose([
    transforms.Resize((64, 256)),
    transforms.ToTensor(),
])

def visualize_attention(model, encoder_outputs, attention, image_filename, ground_truth_latex, show=True):
    """
    Visualizes the attention on the image, highlighting regions with high attention weights.
    """
    attention_map = attention.squeeze(0).cpu().numpy()  # (seq_len, height, width)
    # You may need to adjust the dimensions depending on how attention is represented
    attention_map = np.mean(attention_map, axis=0)  # Mean across the sequence length if needed
    attention_map = np.clip(attention_map, 0, 1)  # Normalize attention map to [0, 1]
    
    # Load the image
    img = Image.open(os.path.join(image_path, image_filename + ".bmp"))
    img = img.convert('RGB')
    img = np.array(img)

    # Overlay attention map on the image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.imshow(attention_map, cmap='Reds', alpha=0.5)  # Red highlights for attention

    # Show ground truth LaTeX expression below the image
    ax.axis('off')
    plt.title(f"Ground Truth: {ground_truth_latex}", fontsize=14)

    if show:
        plt.show()

# Example usage: Visualize the first 5 images
for i, (image_filename, ground_truth_latex) in enumerate(captions.items()):
    if i >= 5:  # Visualize only the first 5 images
        break

    # Assuming the model is already running and you have the encoder outputs and attention
    # Get a dummy batch input (You need to prepare the input for the decoder)
    input_tokens = torch.tensor([1] * 10).unsqueeze(0).long()  # Example, replace with actual input
    previous_attention = torch.zeros(1, 10).long()  # Example, replace with actual attention
    Wpa = torch.zeros(1, 10).long()  # Example, replace with actual pre-aware attention matrix

    # Forward pass through the decoder
    outputs, updated_attention = model(input_tokens, encoder_outputs, previous_attention, Wpa)

    # Visualize the attention
    visualize_attention(model, encoder_outputs, updated_attention, image_filename, ground_truth_latex)
