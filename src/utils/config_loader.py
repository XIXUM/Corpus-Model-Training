import yaml
import torch
import os

def load_config(config_path="config/training_config.yaml"):
    """
    Loads the YAML configuration file.
    """
    if not os.path.exists(config_path):
        # Return defaults if config missing
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        return {
            "hardware": {"device": "cpu"},
            "training": {"epochs": 5, "batch_size": 32, "learning_rate": 0.001}
        }
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_device(config):
    """
    Returns the torch device based on configuration and availability.
    """
    target = config.get("hardware", {}).get("device", "cpu").lower()
    
    if target == "cuda":
        if torch.cuda.is_available():
            print("✅ Using CUDA (Nvidia) hardware.")
            return torch.device("cuda")
        else:
            print("⚠️ CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
            
    elif target == "mps":
        if torch.backends.mps.is_available():
            print("✅ Using MPS (Apple Silicon) hardware.")
            return torch.device("mps")
        else:
            print("⚠️ MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")
            
    else:
        print("ℹ️ Using CPU.")
        return torch.device("cpu")

