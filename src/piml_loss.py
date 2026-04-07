import torch
import torch.nn as nn

class PIMLLoss(nn.Module):
    def __init__(self, lambda_physics=0.1):
        """
        lambda_physics: how strongly to enforce the physics constraint
        0.1 means physics penalty counts 10% as much as prediction error
        """
        super(PIMLLoss, self).__init__()
        self.lambda_physics = lambda_physics
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets, tyre_lives):
        """
        predictions: model's predicted lap times (normalized)
        targets: actual lap times (normalized)
        tyre_lives: tyre age for each lap in the sequence
        """
        # Standard prediction error — how wrong are we?
        prediction_loss = self.mse(predictions, targets)
        
        # Physics penalty — are we violating monotonic degradation?
        # predictions shape: (batch_size,)
        # We need to check consecutive predictions in each batch
        
        physics_penalty = torch.tensor(0.0)
        
        if len(predictions) > 1:
            # Difference between consecutive predictions
            pred_diff = predictions[1:] - predictions[:-1]
            tyre_diff = tyre_lives[1:] - tyre_lives[:-1]
            
            # Find cases where tyre life increased but predicted lap time decreased
            # This violates physics — tyre should be slower as it ages
            violations = torch.clamp(-pred_diff * (tyre_diff > 0).float(), min=0)
            
            physics_penalty = violations.mean()
        
        # Total loss combines both terms
        total_loss = prediction_loss + self.lambda_physics * physics_penalty
        
        return total_loss, prediction_loss, physics_penalty 