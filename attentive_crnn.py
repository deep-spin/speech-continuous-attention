import torch
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from speechbrain.dataio.dataio import length_to_mask

from continuous_attention import ContinuousAttention
from continuous_encoders import ConvEncoder, DiscreteAttentionEncoder
from discrete_attention import DiscreteAttention
from initialization import init_xavier
from scorer import SelfAdditiveScorer


class AttentiveCRNN(torch.nn.Module):

    def __init__(
        self,
        input_size=None,
        cnn_blocks=1,
        cnn_channels=(128,),
        cnn_kernelsize=(3,),
        cnn_activation=torch.nn.ReLU,
        cnn_dropout=0.15,
        cnn_pooling_size=(2,),
        cnn_use_batch_norm=True,
        rnn_class=torch.nn.GRU,
        rnn_neurons=512,
        rnn_bidirectional=True,
        rnn_dropout=0.0,
        attn_domain='discrete',
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
        attn_device=None
    ):
        super().__init__()

        # N blocks of conv + batchnorm + maxpool + relu + dropout
        self.net_convs = torch.nn.Sequential()
        for i in range(cnn_blocks):
            self.net_convs.add_module(
                "conv_{}".format(i),
                torch.nn.Conv1d(
                    input_size if i == 0 else cnn_channels[i-1],
                    cnn_channels[i],
                    kernel_size=cnn_kernelsize[i],
                    padding=cnn_kernelsize[i]//2
                )
            )
            if cnn_use_batch_norm:
                self.net_convs.add_module("bn_{}".format(i), torch.nn.BatchNorm1d(cnn_channels[i]))
            self.net_convs.add_module("maxp_{}".format(i), torch.nn.MaxPool1d(kernel_size=cnn_pooling_size[i],
                                                                              stride=1,
                                                                              padding=cnn_pooling_size[i]//2))
            self.net_convs.add_module("act_{}".format(i), cnn_activation())
            self.net_convs.add_module("drop_{}".format(i), torch.nn.Dropout(cnn_dropout))

        if cnn_blocks == 0:
            cnn_channels = [input_size]

        # a single (bidir) RNN
        if rnn_neurons > 0:
            self.rnn = rnn_class(cnn_channels[-1], rnn_neurons, bidirectional=rnn_bidirectional, batch_first=True)
            self.rnn_dropout = torch.nn.Dropout(rnn_dropout)
            attn_vector_size = rnn_neurons * 2 if rnn_bidirectional else rnn_neurons
        else:
            self.rnn = None
            self.rnn_dropout = None
            attn_vector_size = cnn_channels[-1]

        # attention
        if attn_domain == 'continuous':
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
            self.attn = ContinuousAttention(
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
                vector_size=attn_vector_size,
                gpu_id=attn_device,
            )
        else:
            scorer = SelfAdditiveScorer(attn_vector_size, attn_hidden_size, scaled=False)
            self.attn = DiscreteAttention(
                scorer,
                dropout=attn_dropout,
                max_activation=attn_max_activation
            )

        # Initialize weights properly
        self.init_weights()
        self.conv_out = None

    def init_weights(self):
        # init_xavier(self.net_convs, dist='uniform')
        # init_xavier(self.rnn, dist='uniform')
        # init_xavier(self.attn, dist='uniform')
        pass

    def forward(self, x, lengths=None):
        L = x.shape[1]
        long_lengths = (lengths * L).long()
        mask = length_to_mask(long_lengths, max_len=L, device=x.device)

        # (bs, ts, num_mels) -> (bs, ts, num_filters)
        x = x.transpose(1, 2)
        x = self.net_convs(x)
        x = x.transpose(1, 2)
        self.conv_out = x.detach().cpu()

        if self.rnn is not None:
            # (bs, ts, num_filters) -> (bs, ts, rnn_size)
            x = pack(x, long_lengths.cpu(), batch_first=True, enforce_sorted=False)
            x, hidden = self.rnn(x)
            x, _ = unpack(x, batch_first=True)
            x = self.rnn_dropout(x)

        # (bs, ts, rnn_size) -> (bs, 1, rnn_size)
        x, _ = self.attn(x, x, values=x, mask=mask)

        return x


class Classifier(torch.nn.Module):
    def __init__(self, input_size=None, out_neurons=10):
        super().__init__()
        self.linear_out = torch.nn.Linear(input_size, out_neurons)

    def forward(self, x, lengths=None):
        out = self.linear_out(x)
        return torch.log_softmax(out, dim=-1)
