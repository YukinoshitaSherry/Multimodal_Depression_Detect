import torch.nn as nn
import torch

class BiLSTM_Attention(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=0.5
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # x: (seq_len, batch, input_dim)
        outputs, _ = self.bilstm(x)
        weights = torch.softmax(self.attention(outputs), dim=0)
        return torch.sum(weights * outputs, dim=0)

class GRU_Model(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.5
        )
    
    def forward(self, x):
        _, h_n = self.gru(x)
        return h_n[-1]  # 取最后一层隐藏状态

class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = BiLSTM_Attention()
        self.audio_model = GRU_Model()
        self.fc = nn.Sequential(
            nn.Linear(256+256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, audio_input, text_input):
        audio_feat = self.audio_model(audio_input)
        text_feat = self.text_model(text_input)
        fused = torch.cat([audio_feat, text_feat], dim=1)
        return self.fc(fused)