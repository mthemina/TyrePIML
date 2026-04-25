import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Injects position information into the sequence.
    The Transformer has no inherent sense of order — this fixes that.
    """
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)] # type: ignore
        return self.dropout(x)


class TyreTransformer(nn.Module):
    """
    Transformer encoder for tyre degradation prediction.
    
    Architecture:
    - Input projection: maps 9 features → d_model dimensions
    - Positional encoding: tells the model which lap is which
    - 2 Transformer encoder layers with multi-head self-attention
    - Output head: maps final representation → lap time prediction
    
    Why Transformer > LSTM for this task:
    - Self-attention can directly compare any two laps in the sequence
    - LSTM must pass information through hidden state sequentially
    - Transformer captures long-range dependencies better
    - More parallelizable during training
    """
    def __init__(self, input_size=9, d_model=64, nhead=4,
                 num_layers=2, dropout=0.3):
        super().__init__()

        # Project raw features into model dimension
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True  # (batch, seq, features)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Dropout for MC uncertainty
        self.dropout = nn.Dropout(p=dropout)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_projection(x)       # → (batch, seq, d_model)
        x = self.pos_encoding(x)           # add position info
        x = self.transformer(x)            # self-attention
        x = self.dropout(x[:, -1, :])      # take last timestep
        out = self.output_head(x)          # → (batch, 1)
        return out.squeeze()


if __name__ == '__main__':
    # Sanity check
    model = TyreTransformer(input_size=9)
    x = torch.randn(32, 5, 9)  # batch=32, seq=5, features=9
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}") 