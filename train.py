import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, recall_score, precision_score
import torch
from torch.utils.data import Dataset, DataLoader
from model import FusionModel  # 沿用之前定义的模型

class DepressionDataset(Dataset):
    def __init__(self, data):
        self.audio = [torch.tensor(x) for x in data["audio"]]
        self.text = [torch.tensor(x) for x in data["text"]]
        self.labels = torch.tensor(data["label"].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.audio[idx], self.text[idx], self.labels[idx]

def main():
    data = pd.read_csv("processed_features.csv")
    
    # 使用GroupKFold确保同一被试的数据不分到不同集合
    groups = data["participant"].values
    gkf = GroupKFold(n_splits=5)
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(data, groups=groups)):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        
        # 过采样抑郁类样本（被试级别）
        depressed_participants = train_data[train_data["label"] == 1]["participant"].unique()
        oversampled = train_data[train_data["participant"].isin(depressed_participants)]
        train_data = pd.concat([train_data, oversampled])
        
        # 构建DataLoader
        train_set = DepressionDataset(train_data)
        test_set = DepressionDataset(test_data)
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        
        # 初始化模型
        model = FusionModel()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # 训练
        for epoch in range(50):
            for audio, text, labels in train_loader:
                preds = model(audio, text)
                loss = criterion(preds.squeeze(), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Fold {fold+1} | Epoch {epoch+1} | Loss: {loss.item():.4f}")
        
        # 评估（被试级别：取同一被试所有样本的平均预测）
        participant_preds = test_data.groupby("participant").apply(lambda x: 
            model(torch.stack(x["audio"].apply(torch.tensor)), 
            torch.stack(x["text"].apply(torch.tensor))).mean().item()
        )
        y_true = test_data.groupby("participant")["label"].first().values
        y_pred = (participant_preds > 0.5).astype(int)
        
        print(f"Fold {fold+1} Metrics:")
        print(f"F1: {f1_score(y_true, y_pred):.2f}")
        print(f"Recall: {recall_score(y_true, y_pred):.2f}")
        print(f"Precision: {precision_score(y_true, y_pred):.2f}")

if __name__ == "__main__":
    main()