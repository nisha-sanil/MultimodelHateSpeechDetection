import torch
from pathlib import Path

# ---------------------------------------------------------------------------- #
#                                 Project Paths                                #
# ---------------------------------------------------------------------------- #
# All local files and caches will be stored on the D: drive as requested.
ROOT_DIR = Path("D:/MultimodelHateSpeechDetection")

# --- Main Directories ---
DATA_DIR = ROOT_DIR / "data"
FEATURES_DIR = ROOT_DIR / "features"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
RESULTS_DIR = ROOT_DIR / "results"

# --- Cache Directories for Local Runs ---
# Instructions in README.md will guide user to set these environment variables.
CACHE_DIR = ROOT_DIR / ".cache"
HF_CACHE_DIR = CACHE_DIR / "huggingface"
TORCH_CACHE_DIR = CACHE_DIR / "torch"

HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------- #
#                                Global Settings                               #
# ---------------------------------------------------------------------------- #
SEED = 42
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------- #
#                              Model Configuration                             #
# ---------------------------------------------------------------------------- #
MODEL_CONFIG = {
    "text": {
        "model_name": "microsoft/deberta-base",
        "max_length": 128,
        "embedding_dim": 768, # For deberta-base
    },
    "image": {
        "model_name": "resnet50", # or "efficientnet_b0"
        "embedding_dim": 2048, # For resnet50
        "image_size": 224,
    },
    "aux": {
        "embedding_dim": 64 # As defined in aux_train.json
    },
    "training": {
        "use_amp": torch.cuda.is_available(), # Use mixed precision only on GPU
    }
}