import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from src import config

class TextEncoder(nn.Module):
    """
    A wrapper for Hugging Face text models (e.g., DeBERTa) to extract embeddings.
    """
    def __init__(self, model_name=None, trainable=False, from_local_path=None):
        """
        Args:
            model_name (str, optional): The name of the Hugging Face model. 
                                        Defaults to the one in config.py.
            trainable (bool): Whether the backbone weights are trainable.
            from_local_path (str, optional): Path to a locally saved model to load from.
        """
        super().__init__()
        model_path = from_local_path or model_name or config.MODEL_CONFIG['text']['model_name']
        
        # Load model configuration to get embedding dimension
        hf_config = AutoConfig.from_pretrained(model_path)
        self.embedding_dim = hf_config.hidden_size

        self.model = AutoModel.from_pretrained(model_path)

        for param in self.model.parameters():
            param.requires_grad = trainable

    def forward(self, input_ids, attention_mask):
        """
        Forward pass to get the [CLS] token embedding.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # We use the embedding of the [CLS] token as the sentence representation
        last_hidden_state = outputs.last_hidden_state
        return last_hidden_state[:, 0, :]