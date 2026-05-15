import torch
import torch.nn as nn
from .attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    """
    PURPOSE:
    Run attention multiple times in parallel (multiple "views").

    KEY IDEAS:
    - Each head asks a different question
    - Heads operate independently
    - Outputs are combined back together
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()

        assert embed_dim % num_heads == 0, \
            "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, V
        # Meaning: learn how to ask questions and represent information
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Final projection to mix head outputs
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        INPUT:
        x : (batch, seq_len, embed_dim)

        OUTPUT:
        (batch, seq_len, embed_dim)
        """

        B, T, E = x.shape

        # STEP 1: Create Q, K, V from input embeddings
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # STEP 2: Split embedding into multiple heads
        # Meaning: parallel attention views
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # STEP 3: Apply scaled dot-product attention
        # Meaning: decide importance + blend information
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask, self.attn_dropout)

        # STEP 4: Merge heads back together
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, T, E)

        # STEP 5: Final linear mixing of all heads
        output = self.out_proj(attn_output)

        return output
