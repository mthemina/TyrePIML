import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.dataset import TyreDataset
from src.model import TyreLSTM

def train():
    # Load dataset
    dataset = TyreDataset()
    
    # Split into 80% training, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    # DataLoaders feed data into the model in batches
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    
    # Build model
    model = TyreLSTM()
    
    # Loss function — measures how wrong the predictions are
    criterion = nn.MSELoss()
    
    # Optimizer — adjusts model weights to reduce loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training on {train_size} sequences, validating on {val_size}\n")
    
    # Training loop
    for epoch in range(20):
        # --- Training ---
        model.train()
        train_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()         # Clear old gradients
            predictions = model(x)        # Forward pass
            loss = criterion(predictions, y)  # Calculate error
            loss.backward()               # Backpropagate
            optimizer.step()              # Update weights
            train_loss += loss.item()
        
        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                predictions = model(x)
                loss = criterion(predictions, y)
                val_loss += loss.item()
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0:
            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            print(f"Epoch {epoch+1:2d}/20 — Train Loss: {avg_train:.4f} — Val Loss: {avg_val:.4f}")
    
    return model

if __name__ == '__main__':
    model = train() 

# Save the trained model weights
    import os
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/tyre_lstm_v1.pt')
    print("\nModel saved to models/tyre_lstm_v1.pt") 