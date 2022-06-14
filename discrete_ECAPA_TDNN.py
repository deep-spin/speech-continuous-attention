"""A popular speaker recognition and diarization model.
Authors
 * Hwidong Na 2020
"""

# import os
from functools import partial

import entmax
import torch  # noqa: F401
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN, \
    AttentiveStatisticsPooling

from scorer import SelfAdditiveScorer

available_max_activations = {
    'softmax': torch.softmax,
    'sparsemax': entmax.sparsemax,
    'entmax15': entmax.entmax15,
    'entmax1333': partial(entmax.entmax_bisect, alpha=1.333)
}


class DiscreteAttentiveStatisticsPooling(AttentiveStatisticsPooling):
    def __init__(self, channels, attention_channels=128, global_context=True,
                 scorer='default', max_activation='softmax'):
        super().__init__(channels, attention_channels, global_context)
        self.scorer = scorer
        self.max_activation = max_activation
        self.alphas = None
        if self.scorer == 'self_add':
            self.add_scorer = SelfAdditiveScorer(
                channels * 3 if global_context else channels,
                attention_channels,
                scaled=False
            )

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape [N, C, L].
        """
        L = x.shape[-1]

        def _compute_statistics(x, m, dim=2, eps=self.eps):
            mean = (m * x).sum(dim)
            std = torch.sqrt(
                (m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps)
            )
            return mean, std

        if lengths is None:
            lengths = torch.ones(x.shape[0], device=x.device)

        # Make binary mask of shape [N, 1, L]
        mask = length_to_mask(lengths * L, max_len=L, device=x.device)
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        if self.global_context:
            # torch.std is unstable for backward computation
            # https://github.com/pytorch/pytorch/issues/4320
            total = mask.sum(dim=2, keepdim=True).float()
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x

        # Apply layers
        if self.scorer == 'default':
            attn = self.conv(self.tanh(self.tdnn(attn)))
        elif self.scorer == 'self_add':
            attn = self.add_scorer(attn, attn)

        # Filter out zero-paddings
        attn = attn.masked_fill(mask == 0, float("-inf"))

        max_activation = available_max_activations[self.max_activation]
        attn = max_activation(attn, dim=2)
        mean, std = _compute_statistics(x, attn)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)
        self.alphas = attn

        return pooled_stats


class DiscreteECAPA_TDNN(ECAPA_TDNN):
    def __init__(
            self,
            input_size,
            device="cpu",
            lin_neurons=192,
            activation=torch.nn.ReLU,
            channels=[512, 512, 512, 512, 1536],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=128,
            res2net_scale=8,
            se_channels=128,
            global_context=True,
            attn_scorer='default',
            attn_max_activation='softmax',
    ):
        super().__init__(
            input_size, device, lin_neurons, activation, channels, kernel_sizes, dilations, attention_channels,
            res2net_scale, se_channels, global_context
        )
        self.asp = DiscreteAttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=global_context,
            scorer=attn_scorer,
            max_activation=attn_max_activation
        )
