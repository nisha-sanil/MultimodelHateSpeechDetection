import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttention(nn.Module):
    """Gated Attention mechanism to weigh modalities."""
    def __init__(self, input_dim):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, num_modalities, embedding_dim)
        # Calculate attention scores
        attn_scores = self.attention_net(x)  # (batch_size, num_modalities, 1)
        attn_weights = F.softmax(attn_scores, dim=1) # (batch_size, num_modalities, 1)
        
        # Apply weights
        weighted_features = x * attn_weights
        return torch.sum(weighted_features, dim=1), attn_weights.squeeze(-1)

class CrossModalAttention(nn.Module):
    """A simple cross-modal attention block (self-attention over modalities)."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch_size, num_modalities, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        # Add & Norm
        return self.norm(x + attn_output)

class FusionModel(nn.Module):
    """
    Multimodal fusion model with modality gating, cross-modal attention, and an MLP head.
    """
    def __init__(self, text_dim, image_dim, aux_dim, hidden_dim, num_classes=1, dropout=0.3):
        super().__init__()
        
        # --- Projection layers to unify dimensions ---
        self.projection_dim = hidden_dim // 2
        self.text_proj = nn.Linear(text_dim, self.projection_dim)
        self.image_proj = nn.Linear(image_dim, self.projection_dim)
        self.aux_proj = nn.Linear(aux_dim, self.projection_dim)

        # --- Learned vectors for missing modalities ---
        self.no_text_vector = nn.Parameter(torch.randn(1, self.projection_dim))
        self.no_image_vector = nn.Parameter(torch.randn(1, self.projection_dim))
        self.no_aux_vector = nn.Parameter(torch.randn(1, self.projection_dim))

        # --- Fusion Architecture ---
        self.cross_modal_attention = CrossModalAttention(self.projection_dim, num_heads=4)
        
        # The input to the MLP will be the concatenated features from all modalities
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.projection_dim * 3),
            nn.Linear(self.projection_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, text_feat, image_feat, aux_feat, has_text, has_image, has_aux):
        batch_size = text_feat.shape[0]

        # --- Project and handle missing modalities ---
        # Use learned "no modality" vector where data is missing
        text_proj = self.text_proj(text_feat) * has_text.unsqueeze(1) + \
                    self.no_text_vector.expand(batch_size, -1) * (1 - has_text.unsqueeze(1))
        
        image_proj = self.image_proj(image_feat) * has_image.unsqueeze(1) + \
                     self.no_image_vector.expand(batch_size, -1) * (1 - has_image.unsqueeze(1))

        aux_proj = self.aux_proj(aux_feat) * has_aux.unsqueeze(1) + \
                   self.no_aux_vector.expand(batch_size, -1) * (1 - has_aux.unsqueeze(1))

        # --- Cross-Modal Attention ---
        # Stack modalities for attention: (batch_size, num_modalities, proj_dim)
        modalities = torch.stack([text_proj, image_proj, aux_proj], dim=1)
        attended_modalities = self.cross_modal_attention(modalities)

        # --- Concatenate and Classify ---
        # Flatten the attended features back into a single vector per sample
        fused_features = attended_modalities.view(batch_size, -1)
        
        logits = self.classifier(fused_features)
        
        # We don't return attention weights here, but could for interpretability
        return logits, None