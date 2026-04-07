import torch
import torch.nn as nn

class TyreLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        """
        input_size: number of features per lap (5)
        hidden_size: how many neurons in the LSTM (64 is a good starting point)
        num_layers: how many LSTM layers stacked on top of each other
        """
        super(TyreLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # input shape is (batch, sequence, features)
            dropout=0.2        # randomly disable 20% of neurons to prevent overfitting
        )
        
        # Final layer that converts LSTM output to a single lap time prediction
        self.output_layer = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Run sequence through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Take only the last timestep's output
        last_output = lstm_out[:, -1, :]
        
        # Predict lap time
        prediction = self.output_layer(last_output)
        return prediction.squeeze()