"""A popular speaker recognition and diarization model.
Authors
 * Hwidong Na 2020
"""

# import os
import torch  # noqa: F401
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.dataio.dataio import length_to_mask
from speechbrain.lobes.models.ECAPA_TDNN import Conv1d, BatchNorm1d, TDNNBlock, SERes2NetBlock

from continuous_attention import ContinuousAttention
from continuous_encoders import ConvEncoder, DiscreteAttentionEncoder


class ContinuousAttentiveStatisticsPooling(nn.Module):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.
    Arguments
    ---------
    channels: int
        The number of input channels.
    attention_channels: int
        The number of attention channels.
    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> asp_layer = ContinuousAttentiveStatisticsPooling(64)
    >>> lengths = torch.rand((8,))
    >>> out_tensor = asp_layer(inp_tensor, lengths).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 1, 128])
    """

    def __init__(self, channels, attention_channels=128, global_context=True, cont_attn=None):
        super().__init__()

        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)
        self.tanh = nn.Tanh()
        self.conv = Conv1d(
            in_channels=attention_channels, out_channels=channels, kernel_size=1
        )
        self.cont_attn = cont_attn
        self.values_layer = nn.Linear(channels * 3, channels)

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
        # (bs, hdim, ts) -> (bs, ts, hdim)
        attn = attn.transpose(1, 2)
        values = self.values_layer(attn)
        mean, _, std = self.cont_attn(attn, attn, values, mask=mask.squeeze(1), return_var=True)
        pooled_stats = torch.cat((mean, std), dim=-1)
        pooled_stats = pooled_stats.transpose(1, 2)

        # discrete way:
        # (bs, 3*hdim, ts) -> (bs, hdim, ts)
        # attn = self.conv(self.tanh(self.tdnn(attn)))

        # Filter out zero-paddings
        # attn = attn.masked_fill(mask == 0, float("-inf"))

        # attn = F.softmax(attn, dim=2)
        # mean, std = _compute_statistics(x, attn)
        # Append mean and std of the batch
        # pooled_stats = torch.cat((mean, std), dim=1)
        # pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats


class ContinuousECAPA_TDNN(torch.nn.Module):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143).
    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    activation : torch class
        A class for constructing the activation layers.
    channels : list of ints
        Output channels for TDNN/SERes2Net layer.
    kernel_sizes : list of ints
        List of kernel sizes for each layer.
    dilations : list of ints
        List of dilations for kernels in each layer.
    lin_neurons : int
        Number of neurons in linear layers.
    Example
    -------
    >>> input_feats = torch.rand([5, 120, 80])
    >>> compute_embedding = ECAPA_TDNN(80, lin_neurons=192)
    >>> outputs = compute_embedding(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 192])
    """

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
            attn_cont_encoder='discrete_attn',
            attn_hidden_size=128,
            attn_dropout=0.,
            attn_nb_basis=16,
            attn_penalty=0.1,
            attn_gaussian_sigmas=None,
            attn_wave_b=10000,
            attn_max_seq_len=3000,
            attn_use_power_basis=False,
            attn_use_wave_basis=False,
            attn_use_gaussian_basis=True,
            attn_dynamic_nb_basis=False,
            attn_consider_pad=True,
            attn_max_activation='softmax',
            attn_alpha=None,
            attn_fuse_disc_and_cont=False,
            attn_smooth_values=False,
    ):

        super().__init__()
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.blocks = nn.ModuleList()

        # The initial TDNN layer
        self.blocks.append(
            TDNNBlock(
                input_size,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                activation,
            )
        )

        # SE-Res2Net layers
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation,
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
            channels[-1],
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            activation,
        )

        attn_vector_size = channels[-1] * 3 if global_context else channels[-1]
        if attn_cont_encoder == 'conv':
            cont_encoder = ConvEncoder(
                vector_size=attn_vector_size,
                hidden_size=attn_hidden_size,
                pool='max',
                supp_type='pred'
            )
        elif attn_cont_encoder == 'discrete_attn':
            cont_encoder = DiscreteAttentionEncoder(
                vector_size=attn_vector_size,
                hidden_size=attn_hidden_size,
                pool=None,
                supp_type='pred'
            )
        else:
            raise NotImplementedError

        cont_attn = ContinuousAttention(
            cont_encoder,
            dropout=attn_dropout,
            nb_basis=attn_nb_basis,
            penalty=attn_penalty,
            gaussian_sigmas=attn_gaussian_sigmas,
            wave_b=attn_wave_b,
            max_seq_len=attn_max_seq_len,
            use_power_basis=attn_use_power_basis,
            use_wave_basis=attn_use_wave_basis,
            use_gaussian_basis=attn_use_gaussian_basis,
            dynamic_nb_basis=attn_dynamic_nb_basis,
            consider_pad=attn_consider_pad,
            max_activation=attn_max_activation,
            alpha=attn_alpha,
            fuse_disc_and_cont=attn_fuse_disc_and_cont,
            smooth_values=attn_smooth_values,
            vector_size=channels[-1],
            gpu_id=device,
        )
        # Attentive Statistical Pooling
        self.asp = ContinuousAttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=global_context,
            cont_attn=cont_attn
        )
        self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)

        # Final linear transformation
        self.fc = Conv1d(
            in_channels=channels[-1] * 2,
            out_channels=lin_neurons,
            kernel_size=1,
        )

    def forward(self, x, lengths=None):
        """Returns the embedding vector.
        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        # Minimize transpose for efficiency
        x = x.transpose(1, 2)

        xl = []
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x)

        # Attentive Statistical Pooling
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)

        # Final linear transformation
        x = self.fc(x)

        x = x.transpose(1, 2)
        return x
