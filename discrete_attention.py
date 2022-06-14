from functools import partial

import entmax
import torch
from torch import nn


def unsqueeze_as(tensor, as_tensor, dim=-1):
    """Expand new dimensions based on a template tensor along `dim` axis.
    Args:
        Args:
        tensor (torch.Tensor): tensor with shape (bs, ..., d1)
        as_tensor (torch.Tensor): tensor with shape (bs, ..., n, ..., d2)
    Returns:
        torch.Tensor: (bs, ..., 1, ..., d1)
    """
    x = tensor
    while x.dim() < as_tensor.dim():
        x = x.unsqueeze(dim)
    return x


available_max_activations = {
    'softmax': torch.softmax,
    'sparsemax': entmax.sparsemax,
    'entmax15': entmax.entmax15,
    'entmax1333': partial(entmax.entmax_bisect, alpha=1.333)
}


class DiscreteAttention(nn.Module):
    """Generic Attention Implementation.

       1. Use `query` and `keys` to compute scores (energies)
       2. Apply softmax to get attention probabilities
       3. Perform a dot product between `values` and probabilites (outputs)

    Args:
        scorer (quati.modules.Scorer): a scorer object
        dropout (float): dropout rate after softmax (default: 0.)
    """

    def __init__(self, scorer, dropout=0., max_activation='softmax'):
        super().__init__()
        self.scorer = scorer
        self.dropout = nn.Dropout(dropout)
        self.NEG_INF = float('-inf')  # mask attention scores before softmax
        self.max_activation = available_max_activations[max_activation]
        self.alphas = None

    def forward(self, query, keys, values=None, mask=None):
        """Compute the attention between query, keys and values.

        Args:
            query (torch.Tensor): set of query vectors with shape of
                (batch_size, ..., target_len, hidden_size)
            keys (torch.Tensor): set of keys vectors with shape of
                (batch_size, ..., source_len, hidden_size)
            values (torch.Tensor, optional): set of values vectors with
                shape of: (batch_size, ..., source_len, hidden_size).
                If None, keys are treated as values. Default: None
            mask (torch.ByteTensor, optional): Tensor representing valid
                positions. If None, all positions are considered valid.
                Shape of (batch_size, target_len)

        Returns:
            torch.Tensor: combination of values and attention probabilities.
                Shape of (batch_size, ..., target_len, hidden_size)
            torch.Tensor: attention probabilities between query and keys.
                Shape of (batch_size, ..., target_len, source_len)
        """
        if values is None:
            values = keys

        # get scores (aka energies)
        scores = self.scorer(query, keys)

        # mask out scores to infinity before softmax
        if mask is not None:
            # broadcast in keys' timestep dim many times as needed
            mask = unsqueeze_as(mask, scores, dim=-2)
            scores = scores.masked_fill(mask == 0, self.NEG_INF)

        # apply softmax to get probs
        if scores.dim() == 3:
            bs, ts, hdim = scores.shape
            scores = scores.view(bs*ts, hdim)
            p_attn = self.max_activation(scores, dim=-1)
            p_attn = p_attn.view(bs, ts, hdim)
        else:
            p_attn = self.max_activation(scores, dim=-1)

        # apply dropout - used in Transformer (default: 0)
        p_attn = self.dropout(p_attn)

        # dot product between p_attn and values
        o_attn = torch.matmul(p_attn, values)
        # o_attn = torch.einsum('b...ts,b...sm->b...tm', [p_attn, values])

        self.alphas = p_attn
        return o_attn, p_attn

