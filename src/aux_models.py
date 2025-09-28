import torch
import torch.nn as nn

class SarcasmEmotionLoader(nn.Module):
    """
    A simple model to process sarcasm and emotion labels into a unified embedding.
    - Sarcasm is treated as a binary feature.
    - Emotion is treated as a categorical feature, passed through an embedding layer.
    """
    def __init__(self, num_emotions, emotion_embedding_dim, output_dim):
        super().__init__()
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_embedding_dim)
        
        # The input dimension will be the emotion embedding dim + 1 (for sarcasm)
        input_dim = emotion_embedding_dim + 1
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, sarcasm_label, emotion_label):
        """
        Args:
            sarcasm_label (torch.Tensor): Tensor of shape (batch_size, 1) with binary sarcasm labels.
            emotion_label (torch.Tensor): Tensor of shape (batch_size) with integer emotion labels.
        """
        sarcasm_feature = sarcasm_label.float()
        emotion_feature = self.emotion_embedding(emotion_label)
        
        combined_features = torch.cat([sarcasm_feature, emotion_feature], dim=1)
        return self.fc(combined_features)