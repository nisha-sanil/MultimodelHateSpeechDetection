import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from PIL import Image
import numpy as np
import pandas as pd

from src import config

class MultimodalHateSpeechDataset(Dataset):
    def __init__(self, df, data_dir, feature_dir, mode='train', use_precomputed=False):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.feature_dir = feature_dir
        self.mode = mode
        self.use_precomputed = use_precomputed

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CONFIG['text']['model_name'])

        # Define image transformations
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(config.MODEL_CONFIG['image']['image_size']),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else: # 'val' or 'test'
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(config.MODEL_CONFIG['image']['image_size']),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item_id = row['id']

        # --- Initialize with zeros/empty tensors ---
        text_input_ids = torch.zeros(config.MODEL_CONFIG['text']['max_length'], dtype=torch.long)
        text_attention_mask = torch.zeros(config.MODEL_CONFIG['text']['max_length'], dtype=torch.long)
        image_tensor = torch.zeros(3, config.MODEL_CONFIG['image']['image_size'], config.MODEL_CONFIG['image']['image_size'])
        
        # Modality presence flags
        has_text = torch.tensor(0.0)
        has_image = torch.tensor(0.0)
        has_aux = torch.tensor(0.0)

        # --- Process Text ---
        if pd.notna(row['text']):
            has_text = torch.tensor(1.0)
            encoding = self.tokenizer(
                row['text'],
                add_special_tokens=True,
                max_length=config.MODEL_CONFIG['text']['max_length'],
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            text_input_ids = encoding['input_ids'].squeeze(0)
            text_attention_mask = encoding['attention_mask'].squeeze(0)

        # --- Process Image ---
        if pd.notna(row.get('img_path')):
            img_path = self.data_dir / row['img_path']
            if img_path.exists():
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = self.transform(image)
                    has_image = torch.tensor(1.0)
                except Exception as e:
                    # If image is corrupt, treat it as missing
                    pass

        # --- Prepare sample dictionary ---
        sample = {
            'id': item_id,
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'image': image_tensor,
            'has_text': has_text,
            'has_image': has_image,
            'has_aux': has_aux, # Will be updated below
        }

        # --- Add label if not in test mode ---
        if 'label' in row:
            sample['label'] = torch.tensor(row['label'], dtype=torch.float)

        # --- Add auxiliary features if they exist ---
        if 'sarcasm' in row:
            sample['sarcasm'] = torch.tensor(row['sarcasm'], dtype=torch.long)
            sample['has_aux'] = torch.tensor(1.0)
        if 'emotion' in row:
            sample['emotion'] = torch.tensor(row['emotion'], dtype=torch.long)
            sample['has_aux'] = torch.tensor(1.0)

        # --- Handle pre-computed features for fusion training ---
        if self.use_precomputed:
            feature_file = self.feature_dir / f"{item_id}.npz"
            if feature_file.exists():
                features = np.load(feature_file)
                if 'text' in features:
                    sample['text_embedding'] = torch.from_numpy(features['text'])
                if 'image' in features:
                    sample['image_embedding'] = torch.from_numpy(features['image'])
                if 'aux' in features:
                    sample['aux_embedding'] = torch.from_numpy(features['aux'])

        return sample