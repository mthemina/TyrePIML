import torch
import torch.nn as nn


class PIMLLoss(nn.Module):
    def __init__(self, lambda_physics=0.1):
        super(PIMLLoss, self).__init__()
        self.lambda_physics = lambda_physics
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets, tyre_lives, 
                abrasiveness=None):
        """
        predictions: model's predicted lap times
        targets: actual lap times
        tyre_lives: tyre age for each lap
        abrasiveness: track abrasiveness 0-1 normalized (optional)
                     higher abrasiveness = stronger physics penalty
        """
        # Ensure correct shapes
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        tyre_lives = tyre_lives.view(-1)
        
        # Standard prediction error
        prediction_loss = self.mse(predictions, targets)
        
        # Physics penalty
        physics_penalty = torch.tensor(0.0)
        
        if predictions.shape[0] > 1:
            pred_diff = predictions[1:] - predictions[:-1]
            tyre_diff = tyre_lives[1:] - tyre_lives[:-1]
            
            # Violations: tyre life went up but lap time went down
            violations = torch.clamp(
                -pred_diff * (tyre_diff > 0).float(), min=0
            )
            
            # Scale penalty by track abrasiveness if provided
            # High abrasiveness tracks should penalise violations more
            if abrasiveness is not None:
                abrasiveness_scalar = abrasiveness.mean().item() \
                                     if hasattr(abrasiveness, 'mean') \
                                     else float(abrasiveness)
                # abrasiveness is 0-1 normalized, scale to 0.5-2.0
                abrasiveness_weight = 0.5 + abrasiveness_scalar * 1.5
                physics_penalty = violations.mean() * abrasiveness_weight
            else:
                physics_penalty = violations.mean()
        
        total_loss = prediction_loss + self.lambda_physics * physics_penalty
        
        return total_loss, prediction_loss, physics_penalty


class ThermalPIMLLoss(nn.Module):
    """
    Extended PIML loss adding thermal constraint.
    Tyre thermal energy must increase monotonically with track temperature
    and tyre age combined — hotter conditions accelerate degradation.
    """
    def __init__(self, lambda_physics=0.1, lambda_thermal=0.05):
        super(ThermalPIMLLoss, self).__init__()
        self.lambda_physics = lambda_physics
        self.lambda_thermal = lambda_thermal
        self.mse = nn.MSELoss()
    
    def forward(self, predictions, targets, tyre_lives,
                track_temps=None, abrasiveness=None):
        
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        tyre_lives = tyre_lives.view(-1)
        
        # Base prediction loss
        prediction_loss = self.mse(predictions, targets)
        
        physics_penalty = torch.tensor(0.0)
        thermal_penalty = torch.tensor(0.0)
        
        if predictions.shape[0] > 1:
            pred_diff = predictions[1:] - predictions[:-1]
            tyre_diff = tyre_lives[1:] - tyre_lives[:-1]
            
            # Monotonic degradation constraint
            violations = torch.clamp(
                -pred_diff * (tyre_diff > 0).float(), min=0
            )
            
            if abrasiveness is not None:
                abrasiveness_scalar = abrasiveness.mean().item() \
                                     if hasattr(abrasiveness, 'mean') \
                                     else float(abrasiveness)
                abrasiveness_weight = 0.5 + abrasiveness_scalar * 1.5
                physics_penalty = violations.mean() * abrasiveness_weight
            else:
                physics_penalty = violations.mean()
            
            # Thermal constraint — hotter track = faster degradation
            # If track temp is high, predicted degradation rate should be higher
            if track_temps is not None:
                track_temps = track_temps.view(-1)
                avg_temp = track_temps.mean()
                
                # Expected degradation rate scales with temperature
                # Above 40°C normalized, degradation should accelerate
                temp_threshold = 0.556  # ~40°C normalized
                if avg_temp > temp_threshold:
                    # Penalise if predictions show flat/negative slope in hot conditions
                    temp_factor = (avg_temp - temp_threshold) * 2.0
                    flat_penalties = torch.clamp(-pred_diff, min=0)
                    thermal_penalty = flat_penalties.mean() * temp_factor
        
        total_loss = (prediction_loss + 
                     self.lambda_physics * physics_penalty +
                     self.lambda_thermal * thermal_penalty)
        
        return total_loss, prediction_loss, physics_penalty, thermal_penalty 