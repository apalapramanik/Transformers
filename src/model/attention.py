import torch
import math


def causal_mask(seq_len, device):
    """
    PURPOSE:
    Prevent a word from seeing future words.

    KEY IDEA:
    - Language models must only look LEFT.
    - Mask blocks attention to positions > current index.
    """

    # Upper-triangular matrix (future positions)
    mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device),
        diagonal=1
    )

    # Convert to boolean mask (True = block)
    return mask.bool()


def scaled_dot_product_attention(Q, K, V, mask=None, attn_dropout=None):
    """
    PURPOSE:
    Compute attention = decide importance + blend information.

    KEY IDEAS:
    - Q (Query): what this position is looking for
    - K (Key): what each position represents
    - V (Value): what information each position provides

    INPUT SHAPES:
    Q, K, V : (batch, heads, seq_len, head_dim)
    """

    # Unpack dimensions for readability
    b, h, t, d = Q.shape

    # STEP 1: Compare queries with all keys
    # Meaning: "Which past positions matter?"
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # STEP 2: Scale scores for numerical stability
    # Meaning: "Keep values in a reasonable range"
    scores = scores / math.sqrt(d)

    # STEP 3: Apply causal mask (block future information)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    # STEP 4: Convert scores to attention weights
    # Meaning: "Turn importance into probabilities"
    attn_weights = torch.softmax(scores, dim=-1)
    if attn_dropout is not None:
        attn_weights = attn_dropout(attn_weights)

    # STEP 5: Blend information from important positions
    # Meaning: "Collect values using attention weights"
    output = torch.matmul(attn_weights, V)

    # output: context-aware representations
    # attn_weights: who attended to whom
    return output, attn_weights
