import torch
import torch.nn as nn
import torchvision.models as models

from src import config

class ImageEncoder(nn.Module):
    """
    A wrapper for torchvision image models (e.g., ResNet50) to extract features.
    """
    def __init__(self, model_name=None, pretrained=True, trainable=False):
        """
        Args:
            model_name (str, optional): The name of the torchvision model. 
                                        Defaults to the one in config.py.
            pretrained (bool): Whether to load pretrained weights.
            trainable (bool): Whether the backbone weights are trainable.
        """
        super().__init__()
        model_name = model_name or config.MODEL_CONFIG['image']['model_name']
        
        if model_name == "resnet50":
            base_model = models.resnet50(pretrained=pretrained)
            self.embedding_dim = base_model.fc.in_features
            # Remove the final classification layer
            self.model = nn.Sequential(*list(base_model.children())[:-1])
        else:
            raise NotImplementedError(f"Image model '{model_name}' not supported yet.")

        for param in self.model.parameters():
            param.requires_grad = trainable

    def forward(self, image):
        # The output of the sequential model is (batch_size, embedding_dim, 1, 1)
        return torch.flatten(self.model(image), 1)