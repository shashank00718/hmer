import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision.models import densenet121
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_iterator import dataIterator  # Custom data loading module
from encoder import DenseMD  # Custom DenseNet implementation
from decoder import AttnDecoderCausal  # Custom Attention Decoder


# Configuration
class Config:
    # Training Hyperparameters
    BATCH_SIZE = 4  # Smaller batch size for better generalization
    TEST_BATCH_SIZE = 4
    LEARNING_RATE = 0.0003  # Lower learning rate for better convergence
    MAX_EPOCHS = 500  # More epochs for thorough training
    HIDDEN_SIZE = 512  # Increased model capacity
    TEACHER_FORCING_RATIO = 0.9  # More teacher forcing for stable training
    MAX_SEQUENCE_LENGTH = 150  # Increased to handle longer formulas
    MAX_IMAGE_SIZE = 100000  # Increased to handle larger images
    BATCH_IMAGESIZE = 400000  # Adjusted for new batch size
    
    # Optimization parameters
    NUM_WORKERS = 4  # Reduced to prevent memory issues
    PIN_MEMORY = True
    GRADIENT_CLIP = 2.0  # More conservative gradient clipping
    WEIGHT_DECAY = 0.0001  # L2 regularization
    
    # Learning rate scheduling
    LR_SCHEDULER = True
    LR_STEP_SIZE = 30
    LR_GAMMA = 0.95
    WARMUP_EPOCHS = 5
    MIN_LR = 1e-6
    
    # Dropout and regularization
    DROPOUT = 0.3
    ENCODER_DROPOUT = 0.2
    DECODER_DROPOUT = 0.2
    
    # Model architecture
    ENCODER_LAYERS = 4  # More dense blocks
    DECODER_BLOCKS = 4  # More decoder blocks
    ATTENTION_HEADS = 8  # Multi-head attention
    
    # Image parameters
    IMAGE_HEIGHT = 64  # Increased image height for better detail
    IMAGE_WIDTH = None
    
    # Training optimizations
    USE_AMP = True  # Keep mixed precision
    ACCUMULATION_STEPS = 4  # Effective batch size of 16
    
    # Paths (replace with your actual paths or use environment variables)
    TRAIN_IMAGE_PATH = r'C:\Users\shash\PycharmProjects\hmer\mathwriting-2024-excerpt\train\extracted_images'
    TRAIN_CAPTION_PATH = r'C:\Users\shash\PycharmProjects\hmer\mathwriting-2024-excerpt\train\labels'
    TEST_IMAGE_PATH = r'C:\Users\shash\PycharmProjects\hmer\mathwriting-2024-excerpt\test\extracted_images'
    TEST_CAPTION_PATH = r'C:\Users\shash\PycharmProjects\hmer\mathwriting-2024-excerpt\test\labels'
    DICTIONARY_PATH = r'C:\Users\shash\PycharmProjects\hmer\dictionary.txt'
    PRETRAINED_DENSENET = r'C:\Users\shash\PycharmProjects\hmer\densenet121_pretrained.pth'

    # GPU Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GPU_IDS = [0] if torch.cuda.is_available() else []
    
    # Added new parameter
    USE_PIN_MEMORY = torch.cuda.is_available()


def compute_wer(label, rec):
    """Compute Word Error Rate (WER)"""
    dist_mat = np.zeros((len(label) + 1, len(rec) + 1), dtype='int32')
    dist_mat[0, :] = range(len(rec) + 1)
    dist_mat[:, 0] = range(len(label) + 1)

    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i - 1, j - 1] + (label[i - 1] != rec[j - 1])
            ins_score = dist_mat[i, j - 1] + 1
            del_score = dist_mat[i - 1, j] + 1
            dist_mat[i, j] = min(hit_score, ins_score, del_score)

    dist = dist_mat[len(label), len(rec)]
    return dist, len(label)


def load_dictionary(dict_file):
    """Load dictionary from file"""
    lexicon = {}
    with open(dict_file, 'r') as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) == 2:
                lexicon[parts[0]] = int(parts[1])
    print(f'Total words/phones: {len(lexicon)}')
    return lexicon


class MathExpressionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = [feat for batch in features for feat in batch]
        self.labels = [lab for batch in labels for lab in batch]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx].float()
        label = self.labels[idx]  # return as list or tensor, handle in collate_fn
        return feature, label


def collate_fn(batch):
    features, labels = zip(*batch)
    features = torch.stack(features)
    labels = [torch.tensor(label, dtype=torch.long) for label in labels]
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=0)
    return features, padded_labels


def create_data_loaders(train_images, train_labels, test_images, test_labels, batch_size, num_workers=4):
    train_dataset = MathExpressionDataset(train_images, train_labels)
    test_dataset = MathExpressionDataset(test_images, test_labels)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=Config.USE_PIN_MEMORY,  # Only use pin_memory if GPU is available
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False  # Only use if we have workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.USE_PIN_MEMORY,  # Only use pin_memory if GPU is available
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, test_loader


def train_epoch(encoder, decoder, train_loader, criterion, encoder_optimizer, decoder_optimizer, config):
    encoder.train()
    decoder.train()
    total_loss = 0.0

    # Add progress bar
    pbar = tqdm(train_loader, desc='Training')

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(config.DEVICE)  # [batch_size, 1, H, W]
        labels = labels.to(config.DEVICE)  # [batch_size, seq_len]
        
        # Forward pass through encoder
        encoder_outputs = encoder(images)  # [batch_size, hidden_size, H', W']
        
        # Prepare encoder outputs for decoder
        B, C, H, W = encoder_outputs.size()
        encoder_outputs = encoder_outputs.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # Initialize attention
        previous_attention = torch.zeros(B, H*W).to(config.DEVICE)
        
        # Get target sequence length from labels
        target_length = labels.size(1)
        decoder_input = torch.zeros(B, target_length).long().to(config.DEVICE)
        decoder_input[:, 0] = labels[:, 0]  # Start with first target token
        
        # Pre-aware attention matrix
        Wpa = torch.eye(config.HIDDEN_SIZE).to(config.DEVICE)
        
        # Forward pass through decoder
        decoder_output, _ = decoder(decoder_input, encoder_outputs, previous_attention, Wpa)
        
        # Calculate loss
        loss = criterion(decoder_output.reshape(-1, decoder_output.size(-1)), 
                        labels.reshape(-1))

        # Backpropagation
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        
        if batch_idx % 10 == 0:  # Reduced logging frequency
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    return total_loss / len(train_loader)


def evaluate(encoder, decoder, test_loader, config):
    encoder.eval()
    decoder.eval()
    total_dist = 0
    total_label = 0
    total_line = 0
    total_line_rec = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            # Forward pass through encoder
            encoder_outputs = encoder(images)
            
            # Prepare encoder outputs for decoder
            B, C, H, W = encoder_outputs.size()
            encoder_outputs = encoder_outputs.permute(0, 2, 3, 1).reshape(B, H*W, C)
            
            # Initialize attention and decoder input
            previous_attention = torch.zeros(B, H*W).to(config.DEVICE)
            decoder_input = torch.zeros(B, 1).long().to(config.DEVICE)
            Wpa = torch.eye(config.HIDDEN_SIZE).to(config.DEVICE)
            
            # Generate sequence
            max_length = config.MAX_SEQUENCE_LENGTH
            predictions = []
            
            for t in range(max_length):
                decoder_output, previous_attention = decoder(
                    decoder_input, encoder_outputs, previous_attention, Wpa
                )
                # Get most likely token
                token = decoder_output.argmax(dim=-1)
                predictions.append(token)
                decoder_input = token
                
                # Stop if all sequences have reached end token
                if (token == 0).all():  # Assuming 0 is end token
                    break
            
            predictions = torch.stack(predictions, dim=1)
            
            # Calculate metrics
            for pred, label in zip(predictions, labels):
                dist, length = compute_wer(label.cpu().numpy(), pred.cpu().numpy())
                total_dist += dist
                total_label += length
                total_line += 1
                if dist == 0:
                    total_line_rec += 1

    wer = float(total_dist) / total_label if total_label > 0 else float('inf')
    sacc = float(total_line_rec) / total_line if total_line > 0 else 0
    return wer, sacc


def main():
    config = Config()
    worddicts = load_dictionary(config.DICTIONARY_PATH)

    train_images, train_labels = dataIterator(
        config.TRAIN_IMAGE_PATH,
        config.TRAIN_CAPTION_PATH,
        worddicts,
        batch_size=config.BATCH_SIZE,
        batch_Imagesize=config.BATCH_IMAGESIZE,
        maxlen=config.MAX_SEQUENCE_LENGTH,
        maxImagesize=config.MAX_IMAGE_SIZE
    )

    test_images, test_labels = dataIterator(
        config.TEST_IMAGE_PATH,
        config.TEST_CAPTION_PATH,
        worddicts,
        batch_size=config.BATCH_SIZE,
        batch_Imagesize=config.BATCH_IMAGESIZE,
        maxlen=config.MAX_SEQUENCE_LENGTH,
        maxImagesize=config.MAX_IMAGE_SIZE
    )

    train_loader, test_loader = create_data_loaders(
        train_images,
        train_labels,
        test_images,
        test_labels,
        batch_size=config.BATCH_SIZE
    )

    encoder = DenseMD(
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        hidden_channels=config.HIDDEN_SIZE
    ).to(config.DEVICE)
    
    # Load pretrained weights and adapt to our architecture
    pretrained_dict = torch.load(config.PRETRAINED_DENSENET)
    encoder_dict = encoder.state_dict()
    
    # Filter out first conv layer and any incompatible layers
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in encoder_dict and 'conv0' not in k}
    
    # Update only the compatible layers
    encoder_dict.update(pretrained_dict)
    encoder.load_state_dict(encoder_dict, strict=False)

    decoder = AttnDecoderCausal(
        hidden_size=config.HIDDEN_SIZE,
        output_size=len(worddicts)  # vocabulary size
    ).to(config.DEVICE)

    encoder = encoder.to(config.DEVICE)
    decoder = decoder.to(config.DEVICE)

    if len(config.GPU_IDS) > 1:
        encoder = DataParallel(encoder, device_ids=config.GPU_IDS)
        decoder = DataParallel(decoder, device_ids=config.GPU_IDS)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=config.LEARNING_RATE, momentum=0.9)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=config.LEARNING_RATE, momentum=0.9)

    criterion = nn.NLLLoss()

    best_sacc = 0.0

    for epoch in range(config.MAX_EPOCHS):
        train_loss = train_epoch(encoder, decoder, train_loader, criterion,
                                 encoder_optimizer, decoder_optimizer, config)

        wer, sacc = evaluate(encoder, decoder, test_loader, config)

        print(f'Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, WER = {wer:.4f}, SACC = {sacc:.4f}')

        if sacc > best_sacc:
            best_sacc = sacc
            torch.save(encoder.state_dict(), 'best_encoder.pth')
            torch.save(decoder.state_dict(), 'best_decoder.pth')

    print(f'Best Sequence Accuracy: {best_sacc:.4f}')


if __name__ == '__main__':
    main()
