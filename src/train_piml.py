import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.dataset import TyreDataset, denormalize, LAP_TIME_MIN, LAP_TIME_MAX
from src.model import TyreLSTM
from src.piml_loss import PIMLLoss

def train_piml():
    # Load dataset
    dataset = TyreDataset()
    
    # Same 80/20 split as baseline for fair comparison
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    
    # Build model — same architecture as baseline, fair comparison
    model = TyreLSTM()
    
    # Physics-informed loss instead of plain MSE
    criterion = PIMLLoss(lambda_physics=0.1)   
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training PIML model on {train_size} sequences\n")
    
    for epoch in range(20):
        # --- Training ---
        model.train()
        train_loss = 0
        train_physics = 0
        
        for x, y in train_loader:
            optimizer.zero_grad()
            predictions = model(x)
            
            # Pass tyre life (first feature of last lap in sequence) for physics check
            tyre_lives = x[:, -1, 0]
            
            loss, pred_loss, physics_penalty = criterion(predictions, y, tyre_lives)
            loss.backward()
            optimizer.step()
            
            train_loss += pred_loss.item()
            train_physics += physics_penalty.item()
        
        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                predictions = model(x)
                tyre_lives = x[:, -1, 0]
                loss, pred_loss, physics_penalty = criterion(predictions, y, tyre_lives)
                val_loss += pred_loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            avg_physics = train_physics / len(train_loader)
            print(f"Epoch {epoch+1:2d}/20 — Train Loss: {avg_train:.4f} — Val Loss: {avg_val:.4f} — Physics Penalty: {avg_physics:.4f}")
    
    return model

if __name__ == '__main__':
    model = train_piml() 

import os
import torch
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/tyre_lstm_piml_v1.pt')
print("\nPIML model saved to models/tyre_lstm_piml_v1.pt") 