import torch
import logging
from pathlib import Path
import argparse

from src import config, utils
from src.aux_models import SarcasmEmotionLoader

def main(args):
    """
    This script demonstrates how to initialize and save the auxiliary model.
    In a real project, this would involve loading a dataset with sarcasm/emotion
    labels and training this small model.

    For our purpose, we just initialize it with pre-defined dimensions and save it.
    The fusion model will then load this pre-trained (or in this case, pre-initialized)
    auxiliary model.
    """
    exp_config = utils.load_config(Path(args.config))
    run_name = f"aux_train_{Path(args.config).stem}"
    utils.setup_logging(config.LOGS_DIR, run_name)
    utils.seed_everything(exp_config.get('seed', config.SEED))

    logging.info("Initializing auxiliary model...")

    model = SarcasmEmotionLoader(
        num_emotions=exp_config['model']['num_emotions'],
        emotion_embedding_dim=exp_config['model']['emotion_embedding_dim'],
        output_dim=exp_config['model']['output_dim']
    )

    output_path = Path(exp_config['output_path'])
    output_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), output_path)
    logging.info(f"Auxiliary model initialized and saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Initialize or train auxiliary models.")
    parser.add_argument("--config", type=str, default="experiments/aux_train.json", help="Path to the auxiliary training JSON config file (relative to project root).")
    args = parser.parse_args()
    main(args)