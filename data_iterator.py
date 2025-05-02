import os
import sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def dataIterator(image_folder, label_folder, dictionary, batch_size, batch_Imagesize, maxlen, maxImagesize):
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')])
    label_files = sorted([f for f in os.listdir(label_folder) if f.endswith('.txt')])

    features = {}
    targets = {}

    def get_transform(img):
        # Calculate width to maintain aspect ratio
        w, h = img.size
        target_height = 32
        target_width = int((w / h) * target_height)
        
        return transforms.Compose([
            transforms.Resize((target_height, target_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

    for img_file, lbl_file in tqdm(zip(image_files, label_files), desc='Loading data'):
        img_path = os.path.join(image_folder, img_file)
        lbl_path = os.path.join(label_folder, lbl_file)

        # Load image and convert to grayscale
        img = Image.open(img_path).convert('L')
        
        # Create transform based on image dimensions
        transform = get_transform(img)
        
        # Apply transform
        img_tensor = transform(img)
        features[img_file] = img_tensor

        # Load label
        with open(lbl_path, 'r') as f:
            label_text = f.readline().strip()
            word_list = [dictionary[w] if w in dictionary else 0 for w in label_text.split()]
            targets[img_file] = word_list

    imageSize = {uid: fea.shape[1] * fea.shape[2] for uid, fea in features.items()}
    imageSize = sorted(imageSize.items(), key=lambda d: d[1], reverse=True)

    feature_batch = []
    label_batch = []
    feature_total = []
    label_total = []
    uidList = []

    batch_image_size = 0
    biggest_image_size = 0
    i = 0
    for uid, size in imageSize:
        if size > biggest_image_size:
            biggest_image_size = size
        fea = features[uid]
        lab = targets[uid]
        batch_image_size = biggest_image_size * (i + 1)

        if len(lab) > maxlen or size > maxImagesize:
            continue  # Ignore samples exceeding max length or size

        uidList.append(uid)
        if batch_image_size > batch_Imagesize or i == batch_size:  # Batch is full
            if label_batch:
                feature_total.append(feature_batch)
                label_total.append(label_batch)
            i = 0
            biggest_image_size = size
            feature_batch = []
            label_batch = []
            feature_batch.append(fea)
            label_batch.append(lab)
            batch_image_size = biggest_image_size * (i + 1)
            i += 1
        else:
            feature_batch.append(fea)
            label_batch.append(lab)
            i += 1

    feature_total.append(feature_batch)
    label_total.append(label_batch)

    print('Total ', len(feature_total), 'batch data loaded')
    return feature_total, label_total