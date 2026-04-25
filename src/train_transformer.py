import torch
from torch.utils.data import DataLoader, random_split
from src.dataset import TyreDataset, denormalize, LAP_TIME_MIN, LAP_TIME_MAX
from src.transformer_model import TyreTransformer
from src.piml_loss import ThermalPIMLLoss
import numpy as np


def train_transformer(epochs=30):
    dataset = TyreDataset(use_weather=True, use_track=True)
    input_size = dataset.get_input_size()

    torch.manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    model = TyreTransformer(input_size=input_size, d_model=64, nhead=4, num_layers=2)
    criterion = ThermalPIMLLoss(lambda_physics=0.1, lambda_thermal=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Training Transformer on {train_size} sequences\n")

    best_val = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            predictions = model(x)
            tyre_lives = x[:, -1, 0]
            track_temps = x[:, -1, 5] if x.shape[2] >= 6 else None
            abrasiveness = x[:, -1, 7] if x.shape[2] >= 8 else None
            loss, pred_loss, _, _ = criterion(
                predictions, y, tyre_lives, track_temps, abrasiveness
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += pred_loss.item()

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
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), 'models/tyre_transformer_v1.pt')

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} — "
                  f"Train: {avg_train:.4f} Val: {avg_val:.4f} "
                  f"{'★ Best' if avg_val == best_val else ''}")

    print(f"\nBest val loss: {best_val:.4f}")
    print("Saved to models/tyre_transformer_v1.pt")
    return model


if __name__ == '__main__':
    train_transformer() 