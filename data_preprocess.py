import os
import pandas as pd
import librosa
import numpy as np
import torch
from scipy.signal import find_peaks
from transformers import AutoTokenizer, AutoModel

def extract_pitch_features(audio, sr):
    """提取基频（F0）相关特征"""
    # 使用librosa提取基频
    f0, voiced_flag, voiced_probs = librosa.pyin(audio, 
                                                fmin=librosa.note_to_hz('C2'),
                                                fmax=librosa.note_to_hz('C7'),
                                                sr=sr)
    
    # 计算基频统计特征
    f0_mean = np.nanmean(f0[voiced_flag])
    f0_std = np.nanstd(f0[voiced_flag])
    f0_range = np.nanmax(f0[voiced_flag]) - np.nanmin(f0[voiced_flag])
    
    return [f0_mean, f0_std, f0_range]

def extract_energy_features(audio, sr):
    """提取能量相关特征"""
    # 计算短时能量
    frame_length = int(0.025 * sr)  # 25ms帧长
    hop_length = int(0.010 * sr)    # 10ms帧移
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # 计算能量统计特征
    energy_mean = np.mean(energy)
    energy_std = np.std(energy)
    energy_range = np.max(energy) - np.min(energy)
    
    return [energy_mean, energy_std, energy_range]

def extract_speech_rate(audio, sr):
    """提取语速相关特征"""
    # 使用librosa的onset检测
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    
    # 计算语速（每秒音节数）
    duration = len(audio) / sr
    speech_rate = len(onset_frames) / duration
    
    return [speech_rate]

def extract_formant_features(audio, sr):
    """提取共振峰特征"""
    # 使用librosa提取共振峰
    formants = librosa.feature.formant(y=audio, sr=sr)
    
    # 计算前两个共振峰的统计特征
    f1_mean = np.mean(formants[0])
    f2_mean = np.mean(formants[1])
    
    return [f1_mean, f2_mean]

def extract_mfcc_features(audio, sr):
    """提取MFCC特征"""
    # 提取13维MFCC特征
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    # 计算MFCC统计特征
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    return np.concatenate([mfcc_mean, mfcc_std]).tolist()

def extract_pause_features(audio, sr):
    """提取停顿特征"""
    # 使用能量阈值检测停顿
    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    # 设置能量阈值
    threshold = np.mean(energy) * 0.1
    pauses = energy < threshold
    
    # 计算停顿统计特征
    pause_ratio = np.sum(pauses) / len(pauses)
    pause_duration = np.mean(np.diff(np.where(pauses)[0])) if np.sum(pauses) > 0 else 0
    
    return [pause_ratio, pause_duration]

def extract_audio_features(audio, sr):
    """整合所有音频特征"""
    features = []
    features.extend(extract_pitch_features(audio, sr))
    features.extend(extract_energy_features(audio, sr))
    features.extend(extract_speech_rate(audio, sr))
    features.extend(extract_formant_features(audio, sr))
    features.extend(extract_mfcc_features(audio, sr))
    features.extend(extract_pause_features(audio, sr))
    return features

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
        
        if not (os.path.exists(audio_path) and os.path.exists(text_path)):
            continue
        
        # 提取音频特征（传统方法）
        audio, sr = librosa.load(audio_path, sr=22050)
        audio_feat = extract_audio_features(audio, sr)
        
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
    
    # 遍历所有被试文件夹（只查找t_开头的文件夹）
    for folder in os.listdir(base_dir):
        if folder.startswith("t_") and os.path.isdir(os.path.join(base_dir, folder)):
            folder_path = os.path.join(base_dir, folder)
            participant_features = process_participant(folder_path)
            all_features.extend(participant_features)
    
    # 保存为CSV
    pd.DataFrame(all_features).to_csv("processed_features.csv", index=False)

if __name__ == "__main__":
    main()