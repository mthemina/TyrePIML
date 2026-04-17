import os
import torch
import logging

from src.model import TyreLSTM 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_best_model(track_name: str, compound: str):
    """
    Hierarchical model router that dynamically infers model architecture 
    from saved weights. Utilizes Apple M2 MPS hardware acceleration.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Hardware acceleration active: Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        logger.warning("MPS not available, falling back to CPU.")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    track_model_path = os.path.join(base_dir, "models", "tracks", f"{track_name.replace(' ', '_')}.pt")
    compound_model_path = os.path.join(base_dir, "models", f"tyre_lstm_{compound.lower()}_v1.pt")
    generic_model_path = os.path.join(base_dir, "models", "tyre_lstm_piml_v2.pt")

    selected_model_path = None
    model_tier = None

    if os.path.exists(track_model_path):
        selected_model_path = track_model_path
        model_tier = f"Track-Specific: {track_name}"
    elif os.path.exists(compound_model_path):
        selected_model_path = compound_model_path
        model_tier = f"Compound-Specific: {compound.upper()}"
        logger.info(f"Track model for {track_name} not found. Falling back to Compound.")
    elif os.path.exists(generic_model_path):
        selected_model_path = generic_model_path
        model_tier = "Generic PIML v2 (118 races)"
        logger.info(f"Compound model for {compound} not found. Falling back to Generic.")
    else:
        raise FileNotFoundError(f"CRITICAL: No models found. Not even the generic fallback at {generic_model_path}")

    try:
        # 1. Load the weights dictionary FIRST
        state_dict = torch.load(selected_model_path, map_location=device)
        
        # 2. Dynamically infer the architecture from the saved tensors
        if 'lstm.weight_ih_l0' in state_dict:
            weight_shape = state_dict['lstm.weight_ih_l0'].shape
            inferred_hidden_size = weight_shape[0] // 4
            inferred_input_size = weight_shape[1]
            
            # Count how many LSTM layers exist in the checkpoint
            inferred_num_layers = 0
            while f'lstm.weight_ih_l{inferred_num_layers}' in state_dict:
                inferred_num_layers += 1
                
            logger.info(f"Auto-detected architecture: {inferred_input_size} inputs, {inferred_hidden_size} hidden, {inferred_num_layers} layers.")
        else:
            raise ValueError("State dict does not contain expected LSTM weights.")

        # 3. Instantiate model with the exact dimensions it was trained on
        model = TyreLSTM(
            input_size=inferred_input_size, 
            hidden_size=inferred_hidden_size, 
            num_layers=inferred_num_layers
        )
        
        # 4. Inject weights and push to M2 GPU
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval() 
        
        logger.info(f"Successfully loaded {model_tier} to {device}")
        return model, model_tier

    except Exception as e:
        logger.error(f"Failed to load weights from {selected_model_path}. Error: {str(e)}")
        raise