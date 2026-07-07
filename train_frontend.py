import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import BertModel
import torchvision.models as models
from dataset_preprocessing import MultimodalMusicDataset

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Multimodal Late-Fusion Architecture
class PerceptionFrontend(nn.Module):
    def __init__(self):
        super(PerceptionFrontend, self).__init__()
        # Text Branch (BERT)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Image Branch (CNN - VGG16)
        self.cnn = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # Remove the final classification layer so we just get the raw features
        self.cnn.classifier = nn.Sequential(*list(self.cnn.classifier.children())[:-1])
        
        # Late-Fusion Layer: Combines BERT features (768) + VGG features (4096)
        self.fusion_layer = nn.Linear(768 + 4096, 256)
        
        # Final Output Layer: Predicts [Valence, Arousal]
        self.output_layer = nn.Linear(256, 2)
        
    def forward(self, input_ids, attention_mask, images):
        # Process Text
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_outputs.pooler_output
        
        # Process Images
        image_features = self.cnn(images)
        
        # Concatenate and Fuse
        combined_features = torch.cat((text_features, image_features), dim=1)
        fused = torch.relu(self.fusion_layer(combined_features))
        
        # Output [Valence, Arousal] targets
        va_coords = self.output_layer(fused)
        return va_coords

# The Training Loop
def main():
    print("Setting up hardware...")
    # Automatically use Apple Silicon GPU (MPS) if available, otherwise CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading 6,012 multimodal samples...")
    full_dataset = MultimodalMusicDataset(csv_file='/Users/merlinsteffi/Documents/MusicGen/multimodal_dataset.csv', img_dir='images/')
    
    # 80/20 Train-Test Split
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Tiny batch size to prevent laptop memory overload during testing
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    print("Initializing Model...")
    model = PerceptionFrontend().to(device)
    criterion = nn.MSELoss() # Best for continuous number prediction
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    print("--- Beginning Epoch 1 Training ---")
    model.train()
    running_loss = 0.0
    
    # Loop through the data
    for i, batch in enumerate(train_loader):
        # Move inputs to device (MPS or CPU)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        targets = batch['va_targets'].to(device)
        
        optimizer.zero_grad()
        
        # Forward Pass
        predictions = model(input_ids, attention_mask, images)
        
        # Calculate Error and Backpropagate
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    
        if (i + 1) % 10 == 0:
            print(f"Batch {i+1}/{len(train_loader)} | Loss: {running_loss/10:.4f}")
            running_loss = 0.0
            
    print("Epoch 1 Complete! The frontend is learning.")

if __name__ == "__main__":
    main()