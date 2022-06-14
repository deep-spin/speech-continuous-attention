import torch
from torch import nn

from sparse_continuous_distributions.basis_function import (PowerBasisFunctions, SineBasisFunctions,
                                                            CosineBasisFunctions, GaussianBasisFunctions)
from sparse_continuous_distributions.continuous_entmax import (ContinuousEntmax, ContinuousSoftmax, ContinuousSparsemax,
                                                               ContinuousBiweight, ContinuousTriweight)
from sparse_continuous_distributions.continuous_sparsemax import ContinuousSparsemax as ContinuousSparsemaxOriginal
from sparse_continuous_distributions.continuous_softmax import ContinuousSoftmax as ContinuousSoftmaxOriginal

available_max_activations = {
    'softmax': ContinuousSoftmax,
    'softmax_original': ContinuousSoftmaxOriginal,
    'sparsemax': ContinuousSparsemax,
    'sparsemax_original': ContinuousSparsemaxOriginal,
    'entmax': ContinuousEntmax,
    'biweight': ContinuousBiweight,
    'triweight': ContinuousTriweight,
}


def add_power_basis_functions(min_d=0, max_d=2, device=None):
    degrees = torch.arange(min_d, max_d + 1, device=device).float().to(device)
    return PowerBasisFunctions(degrees)


def add_wave_basis_functions(nb_basis, wave_b, max_seq_len, device=None):
    # sin/cos basis functions similar to Transformers' positional embeddings
    dims = torch.arange(nb_basis // 2, device=device).float()
    omegas = max_seq_len * 1.0 / (wave_b ** (2 * dims / nb_basis)).to(device)
    return SineBasisFunctions(omegas), CosineBasisFunctions(omegas)


def add_gaussian_basis_functions(nb_basis, sigmas, device=None):
    mu, sigma = torch.meshgrid(
            torch.linspace(0, 1, nb_basis // len(sigmas)), torch.Tensor(sigmas),
    )
    mus = mu.flatten().to(device)
    sigmas = sigma.flatten().to(device)
    return GaussianBasisFunctions(mus, sigmas)


def get_positions(length, consider_pad=True):
    if consider_pad and length > 1:
        # insert positions before 0 and after 1 as safe margins for
        # "pad" values (cases where the supp goes beyond [0, 1])
        pad_margin = 0.5
        if length % 2:
            shift = 1.0 / length
            positions = torch.linspace(
                0 - pad_margin + shift,
                1 + pad_margin - shift,
                2 * length - 1,
                )
        else:
            shift = 1.0 / 2 * length
            positions = torch.linspace(
                0 - pad_margin + shift,
                1 + pad_margin - shift,
                2 * length,
                )
    else:
        shift = 1 / float(2 * length)
        positions = torch.linspace(shift, 1 - shift, length)
    return positions


def calculate_G(psi, length, consider_pad, device, penalty=0.1):
    positions = get_positions(length, consider_pad=consider_pad)
    positions = positions.unsqueeze(1).to(device)

    # stack basis functions for each interval
    all_basis = [
        basis_function.evaluate(positions) for basis_function in psi
    ]
    F = torch.cat(all_basis, dim=-1).t().to(device)
    nb_basis = sum([len(b) for b in psi])
    assert F.size(0) == nb_basis

    # compute G with a ridge penalty
    # penalty = 1 / sqrt(length)
    I = torch.eye(nb_basis).to(device)
    G = F.t().matmul((F.matmul(F.t()) + penalty * I).inverse())

    # filter out rows associated with "pad" positions
    if consider_pad and length > 1:
        if length % 2:
            G = G[((length - 1) // 2) : (-(length - 1) // 2), :]
        else:
            G = G[(length // 2) : -(length // 2), :]
    assert G.size(0) == length

    return G


class ContinuousAttention(nn.Module):
    """Generic ContinuousAttention implementation based on ContinuousSparsemax.

       1. Use `query` and `keys` to compute scores (via an encoder)
       2. Map to a probability distribution
       3. Get the final context vector

    Args:
        encoder (ContinuousEncoder): the encoder for getting `mu` and `sigma_sq`
        dropout (float): dropout rate (default: 0)
        nb_basis (int): number of basis functions (default: 16)
        gaussian_sigmas (list of floats): sigmas for gaussian basis functions
            (default is: [0.1, 0.5])
        wave_b (int): frequency param for sine and cosine waves (default: 10000)
        max_seq_len (int): hypothetical maximum sequence length (default: 3000)
        use_power_basis (bool): whether to use power basis functions
        use_wave_basis (bool): whether to use sine/cosine basis functions
        use_gaussian_basis (bool): whether to use gaussian basis functions
        dynamic_nb_basis (bool): whether to use a dynamic nb of basis functions
            where nb_basis = seq_len. If True, the offline computations will be
            saved in cpu memory, and therefore it will impact the runtime
            performance due to memory transfer between cpu and gpu
        consider_pad (bool): whether to consider "pad" positions and insert safe
            margins into the computation of the value function.
        max_activation (str): which prob. density mapping to use:
            sparsemax (default) or softmax (works only with gaussians for now)
        gpu_id (int): gpu id (default: None)
    """

    def __init__(
        self,
        encoder,
        dropout=0.0,
        nb_basis=16,
        penalty=0.1,
        gaussian_sigmas=None,
        wave_b=10000,
        max_seq_len=3000,
        use_power_basis=False,
        use_wave_basis=False,
        use_gaussian_basis=True,
        dynamic_nb_basis=False,
        consider_pad=True,
        max_activation="sparsemax",
        alpha=1.5,
        fuse_disc_and_cont=True,
        smooth_values=False,
        gpu_id=None,
        vector_size=None
    ):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(p=dropout)
        self.nb_basis = nb_basis
        self.penalty = penalty
        self.gaussian_sigmas = gaussian_sigmas if gaussian_sigmas else [.1, .5]
        self.wave_b = wave_b
        self.max_seq_len = max_seq_len
        self.use_power_basis = use_power_basis
        self.use_wave_basis = use_wave_basis
        self.use_gaussian_basis = use_gaussian_basis
        self.dynamic_nb_basis = dynamic_nb_basis
        self.consider_pad = consider_pad
        self.gpu_id = gpu_id
        self.max_activation = max_activation
        self.fuse_disc_and_cont = fuse_disc_and_cont
        self.smooth_values = smooth_values
        self.vector_size = vector_size

        if not any([use_gaussian_basis, use_power_basis, use_wave_basis]):
            raise Exception("You should use at least one basis function.")

        # stored variables (useful for later)
        self.mu = None
        self.variance = None
        self.sigma_sq = None
        self.values = None
        self.mask = None
        self.val_fn_out = None

        # use basis functions in `psi` to define continuous transformation
        # psi = None for now
        act_kwargs = {'alpha': alpha} if max_activation == 'entmax' else {}
        self.cont_max_activation = available_max_activations[max_activation](psi=None, **act_kwargs)

        # compute G offline for each length up to `max_seq_len`
        self.psis = []
        self.Gs = []
        for length in range(1, self.max_seq_len + 1):
            # get the basis functions for this length
            psi = self.create_psi(length)
            G = calculate_G(psi, length, consider_pad, self.gpu_id, penalty=penalty)
            self.psis.append(psi)
            self.Gs.append(G.cpu() if self.dynamic_nb_basis else G)

        # conv to smooth X
        self.conv = None
        if smooth_values:
            self.conv = nn.Conv1d(vector_size, vector_size, kernel_size=3, padding=3//2)

    def sigma_sq_from_variance(self, variance):
        """Variance as computed by E_p[X^2] - E_p[X]^2"""
        if self.max_activation in ['softmax', 'softmax_original']:
            return variance
        if self.max_activation == 'sparsemax_original':
            return (2. / 3) * (5. * variance) ** (3. / 2)
        else:
            return self.cont_max_activation.kernel.sigma_sq_from_variance(variance)

    def support_size_from_sigma_sq(self, sigma_sq):
        if self.max_activation in ['softmax', 'softmax_original']:
            return torch.ones_like(sigma_sq)  # inf
        elif self.max_activation == 'sparsemax_original':
            return 2 * ((3/2)*sigma_sq)**(1/3)
        else:
            return self.cont_max_activation.kernel.support_size_from_sigma_sq(sigma_sq)

    def sigma_sq_from_support_size(self, supp_size):
        if self.max_activation in ['softmax', 'softmax_original']:
            return torch.ones_like(supp_size) * float('inf')
        elif self.max_activation == 'sparsemax_original':
            return (2/3) * (supp_size / 2) ** 3
        else:
            return self.cont_max_activation.kernel.sigma_sq_from_support_size(supp_size)

    @property
    def support_size(self):
        if self.max_activation in ['softmax', 'softmax_original']:
            return torch.ones_like(self.sigma_sq) * 99999999  # inf
        elif self.max_activation == 'sparsemax_original':
            return 2 * ((3/2)*self.sigma_sq)**(1/3)
        else:
            return self.cont_max_activation.kernel.support_size_from_sigma_sq(self.sigma_sq)
            # return self.cont_max_activation.kernel.support_size()

    def create_psi(self, length):
        psi = []
        if self.use_power_basis:
            psi.append(add_power_basis_functions(min_d=0, max_d=2, device=self.gpu_id))
        if self.use_wave_basis:
            nb_basis = length if self.dynamic_nb_basis else self.nb_basis
            nb_basis = max(2, nb_basis)
            psi.extend(add_wave_basis_functions(nb_basis, self.wave_b, self.max_seq_len, device=self.gpu_id))
        if self.use_gaussian_basis:
            nb_basis = length if self.dynamic_nb_basis else self.nb_basis
            nb_basis = max(2, nb_basis)
            psi.append(add_gaussian_basis_functions(nb_basis, sigmas=self.gaussian_sigmas, device=self.gpu_id))
        return psi

    def value_function(self, values, mask=None):
        # Approximate B * F = values via multivariate regression.
        # Use a ridge penalty. The solution is B = values * G
        lengths = mask.sum(-1).int()
        Gs = [self.Gs[l - 1].to(values.device) for l in lengths]
        G = torch.nn.utils.rnn.pad_sequence(Gs, batch_first=True)
        s_values = values
        if self.smooth_values:
            s_values = values * mask.unsqueeze(-1).float()
            s_values = self.conv(s_values.transpose(-1, -2)).transpose(-1, -2)
            s_values = torch.sigmoid(s_values) * values
        B = s_values.transpose(-1, -2).matmul(G)
        return B

    def score_function(self, query, keys, mask=None):
        self.mu, self.variance, disc_p = self.encoder(query, keys, mask=mask)
        self.variance = torch.clamp(self.variance, min=1e-7)
        self.sigma_sq = self.sigma_sq_from_variance(self.variance)

        # clamp sigma_sq
        # lengths = mask.sum(-1).float()
        # target_avg_l = 20.0
        # target_l = torch.min(lengths, torch.ones_like(lengths) * target_avg_l)
        # target_supp = target_l / lengths
        # sigma_sq_max = self.sigma_sq_from_support_size(target_supp)
        # self.sigma_sq = torch.min(self.sigma_sq, sigma_sq_max)
        self.sigma_sq = torch.clamp(self.sigma_sq, min=1e-12)

        theta = torch.zeros(self.mu.size(0), 2, device=query.device)
        theta[:, 0] = self.mu / self.sigma_sq
        theta[:, 1] = -1.0 / (2.0 * self.sigma_sq)
        return theta, disc_p

    def forward(self, query, keys, values, mask=None, return_var=False):
        """
        Compute attention vector.

        Args:
            query (torch.Tensor): shape of (bs, 1, hdim)
            keys (torch.Tensor): shape of (bs, ts, hdim)
            values (torch.Tensor): shape of (bs, ts, hdim)
            mask (torch.ByteTensor): shape of (bs, ts)

        Returns:
            c: torch.Tensor with shape of (bs, 1, hdim)
            r: torch.Tensor with shape of (bs, 1, nb_basis)
        """
        batch_size = keys.size(0)
        seq_len = keys.size(1)

        # make up a dummy mask
        if mask is None:
            mask = torch.ones(batch_size, seq_len, device=query.device)

        # get `mu` and `sigma` as the canonical parameters `theta`
        # (bs, ts, hdim) -> (bs, 2)
        theta, disc_p_attn = self.score_function(query, keys, mask=mask)

        # map to a probability density over basis functions
        # (bs, 2) -> (bs, nb_basis)
        self.cont_max_activation.psi = [psi.to(keys.device) for psi in self.psis[seq_len - 1]]
        r = self.cont_max_activation(theta)

        # create a time dimension
        # (bs, nb_basis) -> (bs, 1, nb_basis)
        r = r.unsqueeze(1)

        # apply dropout (default:0 - like in Transformer arch)
        r = self.dropout(r)

        # compute B using a multivariate regression
        # (bs, ts, hdim) -> (bs, hdim, nb_basis)
        self.values = values
        self.mask = mask
        B = self.value_function(values, mask=mask)
        self.val_fn_out = B.transpose(-1, -2).detach().cpu()

        # (bs, hdim, nb_basis) * (bs, nb_basis, 1) -> (bs, hdim, 1)
        # get the context vector
        c = torch.matmul(B, r.transpose(-1, -2))

        # put time dimension back in the correct place
        # (bs, hdim, 1) -> (bs, 1, hdim)
        c = c.transpose(-1, -2)

        # in case attention probabilities from a discrete attention are passed
        if disc_p_attn is not None and self.fuse_disc_and_cont is True:
            # compute discrete context vector
            disc_c = torch.matmul(disc_p_attn, values)
            # merge with continuous context vector
            c = c + disc_c

        if return_var:
            B = (B - c.transpose(-1, -2)) ** 2
            var = torch.sqrt(torch.matmul(B, r.transpose(-1, -2)).transpose(-1, -2))
            return c, r, var

        return c, r
