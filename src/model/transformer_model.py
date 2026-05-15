import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding
from .transformer_block import TransformerBlock


class TransformerLanguageModel(nn.Module):
    """
    PURPOSE:
    Predict the next token in a sequence using self-attention.

    ARCHITECTURE:
    tokens → embedding
          → positional encoding
          → transformer blocks
          → linear output head
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        ff_hidden_dim,
        max_len=5000,
        dropout=0.1,
    ):
        super().__init__()

        # Token embedding: map token IDs → vectors
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding: inject word order
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_hidden_dim=ff_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Dropout after embedding + positional encoding
        self.emb_dropout = nn.Dropout(dropout)

        # Final normalization (stability before output)
        self.norm = nn.LayerNorm(embed_dim)

        # Output head: map vectors → vocabulary logits
        # bias=False because weights are tied to token_embedding
        self.output_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight

    def forward(self, x, mask=None):
        """
        INPUT:
        x : (batch, seq_len) token indices

        OUTPUT:
        logits : (batch, seq_len, vocab_size)
        """

        # STEP 1: Convert tokens to embeddings
        x = self.token_embedding(x)

        # STEP 2: Add positional information and apply dropout
        x = self.emb_dropout(self.positional_encoding(x))

        # STEP 3: Apply stacked Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # STEP 4: Normalize final representations
        x = self.norm(x)

        # STEP 5: Predict next token logits
        logits = self.output_head(x)

        return logits
