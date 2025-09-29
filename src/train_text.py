import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
from tqdm import tqdm
import numpy as np
import logging
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from src import config, utils
from src.text_model import TextEncoder
from src.dataset import MultimodalHateSpeechDataset # Re-using for text-only loading

def set_trainable_layers(model, num_unfrozen_layers):
    """
    Controls which layers of the transformer are trainable.
    'None' means all layers are trainable.
    """
    if num_unfrozen_layers is None:
        for param in model.parameters():
            param.requires_grad = True
        logging.info("All model layers are trainable.")
        return
    
    if num_unfrozen_layers == 0:
        logging.info("All backbone layers are frozen.")
        return

    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the pooler and the top `num_unfrozen_layers` of the encoder
    if hasattr(model.model, 'pooler'):
        for param in model.model.pooler.parameters():
            param.requires_grad = True

    if hasattr(model.model, 'encoder'):
        num_layers = len(model.model.encoder.layer)
        unfreeze_from = max(0, num_layers - num_unfrozen_layers)
        for i in range(unfreeze_from, num_layers):
            for param in model.model.encoder.layer[i].parameters():
                param.requires_grad = True
        logging.info(f"Unfroze top {num_layers - unfreeze_from} encoder layers and the pooler.")

def train_one_epoch(model, classifier_head, dataloader, optimizer, scheduler, criterion, scaler, device):
    model.train()
    classifier_head.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Fine-tuning Text Model"):
        optimizer.zero_grad()

        input_ids = batch['text_input_ids'].to(device)
        attention_mask = batch['text_attention_mask'].to(device)
        labels = batch['label'].to(device).unsqueeze(1)

        with autocast(enabled=config.MODEL_CONFIG['training']['use_amp']):
            text_embeddings = model(input_ids, attention_mask)
            logits = classifier_head(text_embeddings)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate_one_epoch(model, classifier_head, dataloader, criterion, device):
    model.eval()
    classifier_head.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Text Model"):
            input_ids = batch['text_input_ids'].to(device)
            attention_mask = batch['text_attention_mask'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)

            with autocast(enabled=config.MODEL_CONFIG['training']['use_amp']):
                text_embeddings = model(input_ids, attention_mask)
                logits = classifier_head(text_embeddings)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.sigmoid(logits) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='macro')
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, f1, acc


def main(args):
    exp_config = utils.load_config(Path(args.config))
    run_name = f"text_train_{Path(args.config).stem}"
    utils.setup_logging(config.LOGS_DIR, run_name)
    utils.seed_everything(exp_config.get('seed', config.SEED))
    device = utils.get_device()

    # --- Data ---
    separator = exp_config.get("csv_separator", ",") # Default to comma
    full_df = pd.read_csv(Path(exp_config['data_path']) / exp_config['train_csv'], sep=separator)
    # Filter for samples that have text
    df = full_df[full_df['text'].notna()].reset_index(drop=True)

    label_column = exp_config.get("label_column", "label")
    # For OLID, we need to map labels to 0 and 1
    if 'subtask_a' in df.columns and label_column == 'subtask_a':
        df['label'] = df['subtask_a'].apply(lambda x: 1 if x == 'OFF' else 0)
        label_column = 'label' # Use the new 'label' column for stratification

    train_df, val_df = train_test_split(df, test_size=exp_config.get('val_split', 0.1), random_state=exp_config.get('seed', config.SEED), stratify=df[label_column])

    train_dataset = MultimodalHateSpeechDataset(train_df, Path(exp_config['data_path']), config.FEATURES_DIR, mode='train', use_precomputed=False)
    val_dataset = MultimodalHateSpeechDataset(val_df, Path(exp_config['data_path']), config.FEATURES_DIR, mode='val', use_precomputed=False)
    train_dataloader = DataLoader(train_dataset, batch_size=exp_config['training']['batch_size'], shuffle=True, num_workers=config.NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, batch_size=exp_config['training']['batch_size'], shuffle=False, num_workers=config.NUM_WORKERS)

    # --- Model ---
    # We train the text encoder with a simple linear head for this task
    # If a pre-trained local path is specified (e.g., from OLID training), load it.
    local_model_path = exp_config.get("local_model_path")
    text_encoder = TextEncoder(trainable=False, from_local_path=local_model_path).to(device) # Start with all layers frozen
    if local_model_path:
        logging.info(f"Loaded text model from local path: {local_model_path}")
    classifier_head = torch.nn.Linear(text_encoder.embedding_dim, 1).to(device)

    # --- Training Setup ---
    # Set initial trainable layers (usually just the head)
    set_trainable_layers(text_encoder, 0)

    # Use different learning rates for the backbone and the new head
    optimizer = AdamW([
        {'params': filter(lambda p: p.requires_grad, text_encoder.parameters()), 'lr': exp_config['training']['lr_backbone']},
        {'params': classifier_head.parameters(), 'lr': exp_config['training']['lr_head']}
    ], weight_decay=exp_config['training']['weight_decay'])

    total_steps = len(train_dataloader) * exp_config['training']['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=exp_config['training'].get('warmup_steps', 0), num_training_steps=total_steps)
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = GradScaler(enabled=config.MODEL_CONFIG['training']['use_amp'])

    # --- Gradual Unfreezing ---
    unfreeze_schedule = {int(k): v for k, v in exp_config['training'].get('unfreeze_schedule', {}).items()}

    best_val_f1 = -1
    output_dir = Path(exp_config['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)

    logging.info("Starting text model fine-tuning...")
    for epoch in range(exp_config['training']['epochs']):
        logging.info(f"--- Epoch {epoch+1}/{exp_config['training']['epochs']} ---")

        # Check if we need to unfreeze more layers at the beginning of this epoch
        if epoch in unfreeze_schedule:
            num_layers = unfreeze_schedule[epoch]
            set_trainable_layers(text_encoder, num_layers)
            # Add newly unfrozen parameters to the optimizer
            optimizer.add_param_group({'params': filter(lambda p: p.requires_grad, text_encoder.parameters()), 'lr': exp_config['training']['lr_backbone']})
            logging.info(f"Optimizer updated with newly trainable layers.")

        train_loss = train_one_epoch(text_encoder, classifier_head, train_dataloader, optimizer, scheduler, criterion, scaler, device)
        val_loss, val_f1, val_acc = evaluate_one_epoch(text_encoder, classifier_head, val_dataloader, criterion, device)

        logging.info(f"Epoch {epoch+1} - Average Training Loss: {train_loss:.4f}")
        logging.info(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            logging.info(f"New best validation F1: {best_val_f1:.4f}. Saving model...")
            text_encoder.model.save_pretrained(output_dir)
            torch.save(classifier_head.state_dict(), output_dir / "classifier_head.pth")
            dataset.tokenizer.save_pretrained(output_dir)

    # --- Save Model ---
    logging.info(f"Training complete. Best model saved to {output_dir} with F1 score: {best_val_f1:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune the text encoder (e.g., DeBERTa).")
    parser.add_argument("--config", type=str, required=True, help="Path to the text training JSON config file.")
    args = parser.parse_args()
    main(args)