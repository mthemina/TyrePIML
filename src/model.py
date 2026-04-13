import torch
import torch.nn as nn


class TyreLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, num_layers=2, dropout=0.3):
        super(TyreLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Explicit dropout layer after LSTM — this is what MC dropout uses
        self.dropout = nn.Dropout(p=dropout)
        
        # Two layer output for better expressiveness
        self.hidden_layer = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(32, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        # Dropout applied here — active during MC sampling
        dropped = self.dropout(last_output)
        hidden = self.relu(self.hidden_layer(dropped))
        prediction = self.output_layer(hidden)
        
        return prediction.squeeze() 