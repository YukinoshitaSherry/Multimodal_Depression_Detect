import os
import pandas as pd
import librosa
import torch
import openl3
from transformers import AutoTokenizer, AutoModel

def read_label(folder_path):
    """读取被试标签：优先使用label.txt，其次new_label.txt"""
    label_path = os.path.join(folder_path, "label.txt")
    if not os.path.exists(label_path):
        label_path = os.path.join(folder_path, "new_label.txt")
    with open(label_path, "r") as f:
        return int(f.read().strip())

def process_participant(participant_dir):
    """处理单个被试文件夹"""
    participant_id = os.path.basename(participant_dir)
    label = read_label(participant_dir)
    
    features = []
    # 遍历所有情绪类别（negative/neutral/positive）
    for emotion in ["negative", "neutral", "positive"]:
        audio_path = os.path.join(participant_dir, f"{emotion}.wav")
        text_path = os.path.join(participant_dir, f"{emotion}.txt")
        
        if not (os.path.exists(audio_path) or not (os.path.exists(text_path))):
            continue
        
        # 提取音频特征（OpenL3）
        audio, sr = librosa.load(audio_path, sr=22050)
        audio_emb, _ = openl3.get_audio_embedding(audio, sr, input_repr="mel128")
        audio_feat = torch.tensor(audio_emb.mean(axis=0)).tolist()
        
        # 提取文本特征（BERT）
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        model = AutoModel.from_pretrained("bert-base-chinese")
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        text_feat = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        
        features.append({
            "participant": participant_id,
            "emotion": emotion,
            "audio": audio_feat,
            "text": text_feat,
            "label": label
        })
    return features

def main():
    base_dir = "EATD-Corpus"
    all_features = []
    
    # 遍历所有被试文件夹（t_1到t_13）
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        all_features.extend(process_participant(folder_path))
    
    # 保存为CSV
    pd.DataFrame(all_features).to_csv("processed_features.csv", index=False)

if __name__ == "__main__":
    main()