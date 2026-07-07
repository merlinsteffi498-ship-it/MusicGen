import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import BertTokenizer
from torchvision import transforms

class MultimodalMusicDataset(Dataset):
    def __init__(self, csv_file, img_dir, max_text_len=64):
        
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_text_len = max_text_len
        
        # Image Preprocessing Transform
        self.image_transforms = transforms.Compose([
            transforms.Resize((224, 224)), # changed this from 150 to 224
            transforms.ToTensor(),
            # Normalizes to standard ImageNet values
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Feature Engineering: Categorical to Continuous Valence-Arousal Mapping
        self.emotion_to_va = {
            'Happy':  [0.8, 0.6],
            'Sad':    [-0.8, -0.6],
            'Neutral': [0.0, 0.0]
        }

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
    
        text_string = str(self.data_frame.iloc[idx]['text_prompt'])
        text_inputs = self.tokenizer(
            text_string,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt"
        )
        
       
        img_name = os.path.join(self.img_dir, self.data_frame.iloc[idx]['image_id'])
        image = Image.open(img_name).convert('RGB')
        image_tensor = self.image_transforms(image)
        
       
        label = self.data_frame.iloc[idx]['emotion_label']
        va_coords = torch.tensor(self.emotion_to_va.get(label, [0.0, 0.0]), dtype=torch.float)
        
        return {
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'image': image_tensor,
            'va_targets': va_coords
        }
if __name__ == "__main__":
    try:
        
        print("Loading dataset")
        dataset = MultimodalMusicDataset(csv_file='/Users/merlinsteffi/Documents/MusicGen/multimodal_dataset.csv', img_dir='images/')
        
        print(f"Dataset loaded with {len(dataset)} samples.")
        
        # Test pulling the very first item
        sample = dataset[0]
        print("\n--- Testing Sample 0 ---")
        print(f"Text tokens shape: {sample['input_ids'].shape}")
        print(f"Image tensor shape: {sample['image'].shape}")
        print(f"Target VA coords:  {sample['va_targets']}")

        
    except Exception as e:
        print(f"Error loading dataset: {e}")