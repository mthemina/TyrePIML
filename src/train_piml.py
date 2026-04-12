import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.dataset import TyreDataset, denormalize, LAP_TIME_MIN, LAP_TIME_MAX
from src.model import TyreLSTM
from src.piml_loss import ThermalPIMLLoss

def train_piml(epochs=30):
    # Load full dataset with weather and track features
    dataset = TyreDataset(use_weather=True, use_track=True)
    input_size = dataset.get_input_size()
    print(f"Input size: {input_size} features per lap")
    
    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    
    # Build model with updated input size
    model = TyreLSTM(input_size=input_size, hidden_size=128, num_layers=2)
    
    # Thermal PIML loss
    criterion = ThermalPIMLLoss(lambda_physics=0.1, lambda_thermal=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5
    )
    
    print(f"\nTraining on {train_size} sequences, validating on {val_size}\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_physics = 0
        train_thermal = 0
        
        for x, y in train_loader:
            optimizer.zero_grad()
            predictions = model(x)
            tyre_lives = x[:, -1, 0]
            
            # Extract track temp and abrasiveness if available
            track_temps = x[:, -1, 5] if x.shape[2] >= 6 else None
            abrasiveness = x[:, -1, 7] if x.shape[2] >= 8 else None
            
            loss, pred_loss, phys, therm = criterion(
                predictions, y, tyre_lives, track_temps, abrasiveness
            )
            loss.backward()
            optimizer.step()
            
            train_loss += pred_loss.item()
            train_physics += phys.item()
            train_thermal += therm.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                predictions = model(x)
                tyre_lives = x[:, -1, 0]
                track_temps = x[:, -1, 5] if x.shape[2] >= 6 else None
                abrasiveness = x[:, -1, 7] if x.shape[2] >= 8 else None
                _, pred_loss, _, _ = criterion(
                    predictions, y, tyre_lives, track_temps, abrasiveness
                )
                val_loss += pred_loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            avg_physics = train_physics / len(train_loader)
            avg_thermal = train_thermal / len(train_loader)
            print(f"Epoch {epoch+1:2d}/{epochs} — "
                  f"Train: {avg_train:.4f} Val: {avg_val:.4f} "
                  f"Physics: {avg_physics:.4f} Thermal: {avg_thermal:.4f}")
            
            # Save best model
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), 'models/tyre_lstm_piml_v2.pt')
    
    print(f"\nBest val loss: {best_val_loss:.4f}")
    print("Model saved to models/tyre_lstm_piml_v2.pt")
    return model

if __name__ == '__main__':
    train_piml() 