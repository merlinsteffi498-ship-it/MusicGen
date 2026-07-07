import os
import pandas as pd


base_dir = '/Users/merlinsteffi/Documents/MusicGen/images'

print("Scanning image folders...")
data = []


for emotion_folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, emotion_folder)

   
    if not os.path.isdir(folder_path):
        continue

 
    label = emotion_folder.capitalize()

   
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            
        
            relative_path = os.path.join(emotion_folder, filename)
            
            
            text_prompt = f"A visual representation capturing the feeling of {label.lower()}."
            
            # Add to our dataset
            data.append({
                'image_id': relative_path,
                'text_prompt': text_prompt,
                'emotion_label': label
            })


df = pd.DataFrame(data)
df.to_csv('multimodal_dataset.csv', index=False)

print(f"Success! multimodal_dataset.csv generated with {len(df)} images.")