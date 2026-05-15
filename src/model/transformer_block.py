import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """
    PURPOSE:
    One full Transformer layer.

    WHAT IT DOES:
    1. Look at past words (attention)
    2. Refine information (feedforward)
    3. Preserve original signal (residuals)
    4. Keep values stable (layer norm)
    """

    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()

        # Multi-head self-attention
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)

        # Feedforward network (token-wise)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim),
        )

        # Normalization layers (stability)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout (regularization)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        INPUT:
        x : (batch, seq_len, embed_dim)

        OUTPUT:
        x : (batch, seq_len, embed_dim)
        """

        # ===== ATTENTION BLOCK =====
        # Look at other words and blend information
        attn_out = self.attention(x, mask)

        # Residual connection + normalization
        x = self.norm1(x + self.dropout(attn_out))

        # ===== FEEDFORWARD BLOCK =====
        # Refine each word independently
        ff_out = self.feedforward(x)

        # Residual connection + normalization
        x = self.norm2(x + self.dropout(ff_out))

        return x
