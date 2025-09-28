import torch
import random
import numpy as np
import logging
import json
from pathlib import Path
import sys
from datetime import datetime

from src import config

def seed_everything(seed=None):
    """
    Set a seed for reproducibility.
    """
    if seed is None:
        seed = config.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Set random seed to {seed}")

def get_device():
    """
    Returns the appropriate device (CUDA or CPU).
    """
    logging.info(f"Using device: {config.DEVICE}")
    return config.DEVICE

def setup_logging(log_dir: Path, run_name: str):
    """
    Configures logging to both console and a file.
    """
    log_dir.mkdir(exist_ok=True, parents=True)
    log_filename = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging setup complete. Log file: {log_dir / log_filename}")

def load_config(config_path: Path):
    """
    Loads a JSON configuration file.
    """
    try:
        with open(config_path, 'r') as f:
            exp_config = json.load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return exp_config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        sys.exit(1)

def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    """Saves model and training parameters at checkpoint."""
    torch.save(state, filename)
    if is_best:
        # You can add logic to copy for a 'model_best.pth.tar'
        logging.info("Saved new best model")

def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model and training parameters from a checkpoint."""
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer
