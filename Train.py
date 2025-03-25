import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset
from data_iterator import dataIterator  # Custom data loading module
from encoder import DenseMD  # Custom DenseNet implementation
from decoder import AttnDecoderCausal  # Custom Attention Decoder


# Configuration
class Config:
    # Training Hyperparameters
    BATCH_SIZE = 4
    TEST_BATCH_SIZE = 4
    LEARNING_RATE = 0.0001
    MAX_EPOCHS = 200
    HIDDEN_SIZE = 256
    TEACHER_FORCING_RATIO = 1.0
    MAX_SEQUENCE_LENGTH = 48
    MAX_IMAGE_SIZE = 100000
    BATCH_IMAGESIZE = 500000

    # Paths (replace with your actual paths or use environment variables)
    TRAIN_IMAGE_PATH = r'C:\Users\shash\Downloads\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\off_image_train\off_image_train.pkl'
    TRAIN_CAPTION_PATH = r'C:\Users\shash\Downloads\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\train_caption.txt'
    TEST_IMAGE_PATH = r'C:\Users\shash\Downloads\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\off_image_test\off_image_test.pkl'
    TEST_CAPTION_PATH = r'C:\Users\shash\Downloads\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\test_caption.txt'
    DICTIONARY_PATH = r'C:\Users\shash\Downloads\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\dictionary.txt'
    PRETRAINED_DENSENET = r'C:\Users\shash\Downloads\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\Pytorch-Handwritten-Mathematical-Expression-Recognition-master\densenet121-a639ec97.pth'

    # GPU Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GPU_IDS = [0] if torch.cuda.is_available() else []


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
        """
        Args:
            features (list): List of feature batches
            labels (list): List of corresponding label batches
        """
        # Flatten the batched data
        self.features = [feat for batch in features for feat in batch]
        self.labels = [lab for batch in labels for lab in batch]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        Convert numpy arrays to torch tensors

        Args:
            idx (int): Index of the sample

        Returns:
            tuple: (image tensor, label tensor)
        """
        # Convert features to torch tensor
        # Assuming features are numpy arrays
        feature = torch.from_numpy(self.features[idx]).float()

        # Convert labels to torch tensor
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return feature, label


def create_data_loaders(train_images, train_labels, test_images, test_labels, batch_size, num_workers=2):
    """
    Create train and test data loaders

    Args:
        train_images (list): Training image features
        train_labels (list): Training labels
        test_images (list): Test image features
        test_labels (list): Test labels
        batch_size (int): Batch size for data loader
        num_workers (int): Number of workers for data loading

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = MathExpressionDataset(train_images, train_labels)
    test_dataset = MathExpressionDataset(test_images, test_labels)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader


def train_epoch(encoder, decoder, train_loader, criterion, encoder_optimizer, decoder_optimizer, config):
    encoder.train()
    decoder.train()
    total_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

        # Forward pass
        encoder_output = encoder(images)
        # Add your decoder logic here similar to the original implementation

        # Compute loss
        loss = criterion(decoder_output, labels)

        # Backward and optimize
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

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
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            # Prediction logic goes here
            # Similar to the original implementation's testing section

    wer = float(total_dist) / total_label
    sacc = float(total_line_rec) / total_line
    return wer, sacc


def main():
    config = Config()

    # Load dictionary
    worddicts = load_dictionary(config.DICTIONARY_PATH)

    # Load data
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

    # Create datasets and dataloaders
    train_dataset = MathExpressionDataset(train_images, train_labels, worddicts)
    test_dataset = MathExpressionDataset(test_images, test_labels, worddicts)

    train_loader, test_loader = create_data_loaders(
        train_images,
        train_labels,
        test_images,
        test_labels,
        batch_size=config.BATCH_SIZE
    )

    # Initialize models
    encoder = DenseNet()
    decoder = AttnDecoderCausal(config.HIDDEN_SIZE, 112)

    # Load pretrained weights
    pretrained_dict = torch.load(config.PRETRAINED_DENSENET)
    encoder_dict = encoder.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
    encoder_dict.update(pretrained_dict)
    encoder.load_state_dict(encoder_dict)

    # Move to GPU
    encoder = encoder.to(config.DEVICE)
    decoder = decoder.to(config.DEVICE)

    if len(config.GPU_IDS) > 1:
        encoder = DataParallel(encoder, device_ids=config.GPU_IDS)
        decoder = DataParallel(decoder, device_ids=config.GPU_IDS)

    # Optimizers
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=config.LEARNING_RATE, momentum=0.9)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=config.LEARNING_RATE, momentum=0.9)

    # Loss function
    criterion = nn.NLLLoss()

    best_sacc = 0.0

    # Training loop
    for epoch in range(config.MAX_EPOCHS):
        train_loss = train_epoch(encoder, decoder, train_loader, criterion,
                                 encoder_optimizer, decoder_optimizer, config)

        wer, sacc = evaluate(encoder, decoder, test_loader, config)

        print(f'Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, WER = {wer:.4f}, SACC = {sacc:.4f}')

        # Save best model
        if sacc > best_sacc:
            best_sacc = sacc
            torch.save(encoder.state_dict(), 'best_encoder.pth')
            torch.save(decoder.state_dict(), 'best_decoder.pth')

    print(f'Best Sequence Accuracy: {best_sacc:.4f}')


if __name__ == '__main__':
    main()
