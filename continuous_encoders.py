"""
This file contains encoders for the continuous attention module.
The encoders implemented here are not in a continuous domain.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from scorer import SelfAdditiveScorer, OperationScorer


class ContinuousEncoder(nn.Module):
    def forward(self, query, keys, mask=None):
        """
        Encode a query vector and a keys matrix to `mu` and `sigma_sq`.

        Args:
            query (torch.Tensor): shape of (batch_size, 1, hdim)
            keys (torch.Tensor): shape of (batch_size, seq_len, hdim)
            mask (torch.Tensor): shape of (batch_size, seq_len)

        Returns:
            mu (torch.Tensor): shape of (batch_size,)
            sigma_sq (torch.Tensor): shape of (batch_size,)
            p_attn (torch.Tensor): shape of (batch_size, 1, seq_len) if the
                encoder uses a discrete attention - otherwise it is None
        """
        raise NotImplementedError


class LSTMEncoder(ContinuousEncoder):
    """
    Uses a BiLSTM + Average states + Linear or
    BiLSTM + Last hidden state + Linear to get `mu` and `sigma_sq`
    """
    def __init__(self, vector_size, hidden_size, pool='last', supp_type='pred'):
        super().__init__()
        self.lstm = nn.LSTM(input_size=vector_size, hidden_size=hidden_size,
                            bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2 * hidden_size, 2)
        self.pool = pool
        self.supp_type = supp_type

    def forward(self, query, keys, mask=None):
        # if provided, use lengths to ignore pad positions
        lengths = mask.int().sum(dim=-1)

        # apply lstm using packed sequences
        x = pack(keys, lengths, batch_first=True, enforce_sorted=False)
        x, (h, c) = self.lstm(x)
        x, _ = unpack(x, batch_first=True)

        # pool a vector from the lstm states
        if self.pool == 'avg':
            h = (x * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
        elif self.pool == 'max':
            h = x.max(dim=1)[0]
        elif self.pool == 'last':
            # concat forward and backward directions
            # (dirs, bs, hidden_size) -> (bs, 2 * hidden_size)
            h = h.transpose(0, 1).reshape(keys.size(0), -1)

        # use a hidden linear layer (query = param vector)
        # (bs, dirs*hidden_size) -> (bs, 2)
        h = torch.tanh(h)
        h = self.linear(h)

        # predict_mu with a sigmoid activation -> [0, 1]
        mu = torch.sigmoid(h[:, 0])

        if self.supp_type == 'minmax':
            # sigma_sq between 0.01 and 0.04 (hardcoded)
            p_min = 0.01
            p_max = 0.03
            c_min = (2. / 3) * p_min ** 3
            c_max = (2. / 3) * p_max ** 3
            sigma_sq = c_min + (c_max - c_min) * torch.sigmoid(h[:, 1])

        elif self.supp_type == 'const':
            # +/-10 words according to the support formula (hardcoded)
            # supp = lambda sq: (1.5 * sq) ** (1 / 3) * L
            p = 10.
            L = keys.size(1)
            sigma_sq = torch.ones_like(mu) * (2. / 3) * (p / L) ** 3
            # assert torch.allclose(supp(sigma_sq), p)

        else:
            # predict sigma_sq with a softplus activation -> [0, inf]
            sigma_sq = torch.nn.functional.softplus(h[:, 1])

        return mu, sigma_sq, None


class ConvEncoder(ContinuousEncoder):
    """
    Uses a Conv1D + Max over time + Linear to get `mu` and `sigma_sq`.
    """
    def __init__(self, vector_size, hidden_size, pool='max', supp_type='pred'):
        super().__init__()
        self.conv = nn.Conv1d(vector_size, hidden_size, kernel_size=3)
        self.linear = nn.Linear(hidden_size, 2)
        self.scorer = SelfAdditiveScorer(vector_size, hidden_size)
        self.pool = pool
        self.supp_type = supp_type
        self.q = self.k = self.alphas = self.mask = None

    def discrete_attn(self, query, keys, mask=None):
        # get scores (aka energies) -> (bs, 1, ts)
        scores = self.scorer(query, keys)

        # mask out scores to infinity before softmax
        if mask is not None:
            # broadcast in keys' timestep dim (bs, ts) -> (bs, 1, ts)
            mask = mask.unsqueeze(-2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # apply softmax to get probs (bs, 1, ts) -> (bs, 1, ts) in \simplex^ts
        p_attn = torch.softmax(scores, dim=-1)
        self.q = query
        self.k = keys
        self.mask = mask
        self.alphas = p_attn
        return p_attn

    def forward(self, query, keys, mask=None):
        p_attn = self.discrete_attn(query, keys, mask=mask)

        # if provided, use lengths to ignore pad positions
        lengths = mask.int().sum(-1)

        # apply conv properly
        x = keys * mask.unsqueeze(-1).float() if mask is not None else keys
        x = self.conv(x.transpose(-1, -2)).transpose(-1, -2)
        x = torch.relu(x)

        # pool a vector from the lstm states
        if self.pool == 'avg':
            x = (x * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(-1)
        elif self.pool == 'max':
            x = x.max(dim=1)[0]
        elif self.pool == 'last':
            arange = torch.arange(x.shape[0]).to(x.device)
            x = x[arange, lengths - 1].squeeze(1)

        # use a hidden linear layer (query = param vector)
        # (bs, dirs*hidden_size) -> (bs, 2)
        x = self.linear(x)

        # predict_mu with a sigmoid activation -> [0, 1]
        mu = torch.sigmoid(x[:, 0])

        # get sigma_sq according to the supp_type strategy (hardcoded)
        if self.supp_type == 'minmax':
            p_min = 0.01
            p_max = 0.03
            c_min = (2. / 3) * p_min ** 3
            c_max = (2. / 3) * p_max ** 3
            sigma_sq = c_min + (c_max - c_min) * torch.sigmoid(x[:, 1])

        elif self.supp_type == 'const':
            L = keys.size(1)
            p = 10.
            sigma_sq = torch.ones_like(mu) * (2. / 3) * (p / L) ** 3

        else:
            sigma_sq = torch.nn.functional.softplus(x[:, 1])

        return mu, sigma_sq, p_attn


class DiscreteAttentionEncoder(ContinuousEncoder):
    """
    Uses a Discrete Attention (additive) to get `mu` and `sigma_sq` as an
    expectation and variance of the positions ell/L, respectively.
    """
    def __init__(self, vector_size, hidden_size, pool=None, supp_type='pred'):
        super().__init__()
        self.scorer = SelfAdditiveScorer(vector_size, hidden_size)
        self.supp_type = supp_type
        self.pool = pool  # ignored
        self.q = self.k = self.alphas = self.mask = None

    def discrete_attn(self, query, keys, mask=None):
        # get scores (aka energies) -> (bs, 1, ts)
        scores = self.scorer(query, keys)

        # mask out scores to infinity before softmax
        if mask is not None:
            # broadcast in keys' timestep dim (bs, ts) -> (bs, 1, ts)
            mask = mask.unsqueeze(-2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # apply softmax to get probs (bs, 1, ts) -> (bs, 1, ts) in \simplex^ts
        p_attn = torch.softmax(scores, dim=-1)
        self.q = query
        self.k = keys
        self.mask = mask
        self.alphas = p_attn

        return p_attn

    def forward(self, query, keys, mask=None):
        # apply discrete attention and get softmax probabilities
        p_attn = self.discrete_attn(query, keys, mask=mask)

        # each ith entry represents a position in [0, 1] linearly spaced by L
        # L = keys.size(1)
        # positions_old = torch.linspace(0, 1, L, device=keys.device)

        # dynamic positions
        lengths = mask.sum(-1).squeeze().int().tolist()
        if keys.shape[0] == 1:
            lengths = [lengths]
        all_pos = [torch.linspace(0, 1, l, device=keys.device) for l in lengths]
        positions = torch.nn.utils.rnn.pad_sequence(all_pos, batch_first=True)

        # E_p_attn[positions]
        # mu_old = p_attn.matmul(positions_old).squeeze(1)
        mu = (p_attn.squeeze(1) * positions).sum(-1)

        # var[positions]
        # variance_old = p_attn.matmul(positions_old ** 2).squeeze(1) - mu_old ** 2  # noqa
        variance = (p_attn.squeeze(1) * positions ** 2).sum(-1) - mu ** 2

        return mu, variance, p_attn


class LastHiddenStateEncoder(ContinuousEncoder):
    """
    Extremely simple encoder. It assumes that `query` is the last hidden state
    from a previous layer and apply a linear map to get `mu` and `sigma_sq`.
    This class was not used in the experiments for continuous attention.
    """
    def __init__(self, vector_size, supp_type='pred'):
        super().__init__()
        self.linear = nn.Linear(vector_size, 2)
        self.supp_type = supp_type

    def forward(self, query, keys, mask=None):
        # query.shape (bs, 1, hdim) = last hidden state
        # keys.shape (bs, ts, hdim)
        x = query.squeeze(1)
        if query.size(1) > 1:
            lengths = mask.int().sum(-1)
            arange = torch.arange(x.shape[0]).to(x.device)
            x = query[arange, lengths - 1].squeeze(1)
        x = self.linear(x)
        mu = torch.sigmoid(x[:, 0])
        sigma_sq = torch.nn.functional.softplus(x[:, 1])
        return mu, sigma_sq


class PairDiscreteAttentionEncoder(ContinuousEncoder):
    """
    Uses a Discrete Attention (additive) to get `mu` and `sigma_sq` as an
    expectation and variance of the positions ell/L, respectively.
    """
    def __init__(self, vector_size, hidden_size, pool=None, supp_type='pred'):
        super().__init__()
        self.scorer = OperationScorer(vector_size, vector_size, hidden_size, op='add', scaled=False)
        self.supp_type = supp_type
        self.pool = pool  # ignored
        self.q = self.k = self.alphas = self.mask = None

    def discrete_attn(self, query, keys, mask=None):
        # get scores (aka energies) -> (bs, 1, ts)
        scores = self.scorer(query, keys)

        # mask out scores to infinity before softmax
        if mask is not None:
            # broadcast in keys' timestep dim (bs, ts) -> (bs, 1, ts)
            mask = mask.unsqueeze(-2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # apply softmax to get probs (bs, 1, ts) -> (bs, 1, ts) in \simplex^ts
        p_attn = torch.softmax(scores, dim=-1)
        self.q = query
        self.k = keys
        self.mask = mask
        self.alphas = p_attn

        return p_attn

    def forward(self, query, keys, mask=None):
        # apply discrete attention and get softmax probabilities
        p_attn = self.discrete_attn(query, keys, mask=mask)

        # each ith entry represents a position in [0, 1] linearly spaced by L
        # L = keys.size(1)
        # positions_old = torch.linspace(0, 1, L, device=keys.device)

        # dynamic positions
        lengths = mask.sum(-1).squeeze().tolist()
        if keys.shape[0] == 1:
            lengths = [lengths]
        all_pos = [torch.linspace(0, 1, l, device=keys.device) for l in lengths]
        positions = torch.nn.utils.rnn.pad_sequence(all_pos, batch_first=True)

        # E_p_attn[positions]
        # mu_old = p_attn.matmul(positions_old).squeeze(1)
        mu = (p_attn.squeeze(1) * positions).sum(-1)

        # var[positions]
        # variance_old = p_attn.matmul(positions_old ** 2).squeeze(1) - mu_old ** 2  # noqa
        variance = (p_attn.squeeze(1) * positions ** 2).sum(-1) - mu ** 2

        return mu, variance, p_attn
