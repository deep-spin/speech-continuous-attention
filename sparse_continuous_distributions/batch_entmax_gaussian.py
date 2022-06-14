import torch
import math


def _gamma(s):
    return math.gamma(s) #torch.lgamma(torch(s)).exp()

def _radius(n, alpha):
    """Return radius R for a given dimension n and alpha."""
    return ((_gamma(n/2 + alpha/(alpha-1)) /
             (_gamma(alpha/(alpha-1)) * math.pi**(n/2))) *
            (2 / (alpha-1)) ** (1/(alpha-1))) ** ((alpha-1)/(2 + (alpha-1)*n))


class EntmaxGaussian1D(object):
    def __init__(self, alpha, mu=None, sigma_sq=None, support_size=None):
        """Create 1D (2-alpha)-Gaussian with parameter alpha.
        The density is
            p(x) = [(alpha-1)*(-tau - .5*(x-mu)**2/sigma_sq)]_+**(1/(alpha-1)).
        If sigma_sq == None, it can be paremetrized by the support_size instead
        (convenient for uniform distributions, where alpha=inf).
        mu and sigma_sq (or support_size) are tensors with the same dimensions.
        """
        self._alpha = alpha
        self._R = _radius(1, alpha) if alpha != 1 else math.inf
        if mu is not None:
            self.set_parameters(mu, sigma_sq, support_size)

    def set_parameters(self, mu, sigma_sq=None, support_size=None):
        self._mu = mu
        if sigma_sq is None:
            self._a = support_size/2
            self._sigma_sq = self._sigma_sq_from_a(self._a)
            self._tau = self._compute_tau()
        else:
            self._sigma_sq = sigma_sq
            self._tau = self._compute_tau()
            self._a = self._compute_a()

    def _compute_tau(self):
        """Return the threshold tau in the density expression."""
        return (-(self._R**2)/2 * self._sigma_sq **
                (-(self._alpha-1) / (self._alpha+1)))

    def _compute_a(self):
        """Return the value a = |x-mu| where the density vanishes."""
        return torch.sqrt(-2 * self._tau * self._sigma_sq)

    def _sigma_sq_from_a(self, a):
        return (a / self._R) ** (self._alpha+1)

    def _sigma_sq_from_variance(self, variance):
        return ((1 + 2*self._alpha/(self._alpha-1)) / (self._R**2)
                * variance) ** ((self._alpha + 1)/2)

    def _support_size_from_sigma_sq(self, sigma_sq):
        a = torch.sqrt(-2 * self._tau * sigma_sq)
        return 2 * a

    def mean(self):
        return self._mu

    def variance(self):
        if self._alpha == math.inf:
            return self._a**2 / 3
        else:
            # Equivalently (without tau):
            # return ((self._R**2) / (1 + 2*self._alpha/(self._alpha-1)) *
            #        self._sigma_sq ** (2/(self._alpha + 1)))
            return ((-2*self._tau)/(1 + 2*self._alpha/(self._alpha-1)) *
                    self._sigma_sq)

    def support_size(self):
        return 2*self._a

    def pdf(self, x):
        """Return the probability density function value for `x`.
        `x` is an arbitrary tensor whose last dimensions should match those of
        `self._mu` or are singleton.
        The output a tensor with the same dimension as `x`.
        Example:
        >>> entmax = EntmaxGaussian(alpha=1.5, mu=torch.randn(3,4),
                                    sigma_sq=torch.randn(3,4))
        >>> x = torch.randn(2,7,3)
        >>> x = x[(...,) + (None,) * 2]  # dim is now (2,7,3,1,1)
        >>> return entmax.pdf(x)  # dim is (2,7,3,3,4)
        """
        ndims = x.ndim - self._mu.ndim
        if self._alpha == math.inf:
            mu = self._mu[(None,)*ndims]
            a = self._a[(None,)*ndims]
            mask = (x >= mu - a) & (x <= mu + a)
            p = torch.zeros_like(mask).float() + 1/(2*a)
            p[~mask] = 0
            return p
        else:
            mu = self._mu[(None,)*ndims]
            sigma_sq = self._sigma_sq[(None,)*ndims]
            tau = self._tau[(None,)*ndims]
            return torch.clamp((self._alpha-1)*(-tau - .5*(x-mu)**2/sigma_sq),
                               min=0)**(1/(self._alpha-1))

    def sample(self, m):
        """Generate a random sample of size `m`."""
        raise NotImplementedError

    def tsallis_entropy(self):
        """Compute Tsallis alpha-entropy."""
        raise NotImplementedError


class Gaussian1D(object):
    def __init__(self, mu=None, sigma_sq=None):
        """Create 1D beta-Gaussian with alpha=2 (sparsemax)."""
        self._alpha = 1
        if mu is not None:
            self.set_parameters(mu, sigma_sq)

    def set_parameters(self, mu, sigma_sq):
        self._mu = mu
        self._sigma_sq = sigma_sq

    def mean(self):
        return self._mu

    def variance(self):
        return self._sigma_sq

    def pdf(self, x):
        mu = self._mu[(None,)*x.ndim]
        sigma_sq = self._sigma_sq[(None,)*x.ndim]
        x = x[(...,) + (None,) * self._mu.ndim]
        return (1/torch.sqrt(2*math.pi*sigma_sq) *
                torch.exp(-.5*(x-mu)**2/sigma_sq))

    def sample(self, m):
        """Generate a random sample of size `m`."""
        raise NotImplementedError

    def tsallis_entropy(self):
        """Compute Tsallis alpha-entropy."""
        raise NotImplementedError


class SparsemaxGaussian1D(object):
    def __init__(self, mu=None, sigma_sq=None, support_size=None):
        """Create 1D beta-Gaussian with alpha=2 (sparsemax)."""
        self._alpha = 2
        self._R = (3/2)**(1/3)
        if mu is not None:
            self.set_parameters(mu, sigma_sq, support_size)

    def set_parameters(self, mu, sigma_sq=None, support_size=None):
        self._mu = mu
        if sigma_sq is None:
            self._a = support_size/2
            self._sigma_sq = self._sigma_sq_from_a(self._a)
            self._tau = self._compute_tau()
        else:
            self._sigma_sq = sigma_sq
            self._tau = self._compute_tau()
            self._a = self._compute_a()

    def _compute_tau(self):
        return -.5*((3/2)**2/self._sigma_sq)**(1/3)

    def _compute_a(self):
        return ((3/2)*self._sigma_sq)**(1/3)

    def _sigma_sq_from_a(self, a):
        return (2/3) * a**3

    def _sigma_sq_from_variance(self, variance):
        return 2/3 * (5*variance)**(3/2)

    def _support_size_from_sigma_sq(self, sigma_sq):
        a = ((3/2)*sigma_sq)**(1/3)
        return 2 * a

    def mean(self):
        return self._mu

    def variance(self):
        return 1/5 * ((3/2) * self._sigma_sq)**(2/3)

    def support_size(self):
        return 2*self._a

    def pdf(self, x):
        ndims = x.ndim - self._mu.ndim
        mu = self._mu[(None,)*ndims]
        sigma_sq = self._sigma_sq[(None,)*ndims]
        tau = self._tau[(None,)*ndims]
        res = -tau - .5*(x-mu)**2/sigma_sq
        return torch.clamp(res, min=0)

    def sample(self, m):
        """Generate a random sample of size `m`."""
        raise NotImplementedError

    def tsallis_entropy(self):
        """Compute Tsallis alpha-entropy."""
        raise NotImplementedError


class BiweightGaussian1D(object):
    def __init__(self, mu=None, sigma_sq=None, support_size=None):
        """Create 1D beta-Gaussian with alpha=1.5 (biweight)."""
        self._alpha = 1.5
        self._R = _radius(1, self._alpha)  # 15**(1/5)
        if mu is not None:
            self.set_parameters(mu, sigma_sq, support_size)

    def set_parameters(self, mu, sigma_sq=None, support_size=None):
        self._mu = mu
        if sigma_sq is None:
            self._a = support_size/2
            self._sigma_sq = self._sigma_sq_from_a(self._a)
            self._tau = self._compute_tau()
        else:
            self._sigma_sq = sigma_sq
            self._tau = self._compute_tau()
            self._a = self._compute_a()

    def _compute_tau(self):
        return -.5*(15**2/self._sigma_sq)**(1/5)

    def _compute_a(self):
        return (15*self._sigma_sq**2)**(1/5)

    def _sigma_sq_from_a(self, a):
        return (a / self._R) ** (self._alpha+1)

    def _sigma_sq_from_variance(self, variance):
        return (1/15)**(1/2) * (7*variance)**(5/4)

    def _support_size_from_sigma_sq(self, sigma_sq):
        a = (15*sigma_sq**2)**(1/5)
        return 2 * a

    def mean(self):
        return self._mu

    def variance(self):
        return ((-2*self._tau)/(1 + 2*self._alpha/(self._alpha-1)) *
                self._sigma_sq)

    def support_size(self):
        return 2*self._a

    def pdf(self, x):
        ndims = x.ndim - self._mu.ndim
        mu = self._mu[(None,)*ndims]
        sigma_sq = self._sigma_sq[(None,)*ndims]
        tau = self._tau[(None,)*ndims]
        return torch.clamp(.5*(-tau - .5*(x-mu)**2/sigma_sq), min=0)**2

    def sample(self, m):
        """Generate a random sample of size `m`."""
        raise NotImplementedError

    def tsallis_entropy(self):
        """Compute Tsallis alpha-entropy."""
        raise NotImplementedError


class TriweightGaussian1D(object):
    def __init__(self, mu=None, sigma_sq=None, support_size=None):
        """Create 1D beta-Gaussian with alpha=4/3 (triweight)."""
        self._alpha = 4/3
        self._R = _radius(1, self._alpha)  # (945/4)**(1/7)
        if mu is not None:
            self.set_parameters(mu, sigma_sq, support_size)

    def set_parameters(self, mu, sigma_sq=None, support_size=None):
        self._mu = mu
        if sigma_sq is None:
            self._a = support_size/2
            self._sigma_sq = self._sigma_sq_from_a(self._a)
            self._tau = self._compute_tau()
        else:
            self._sigma_sq = sigma_sq
            self._tau = self._compute_tau()
            self._a = self._compute_a()

    def _compute_tau(self):
        return -.5*((945/4)**2/self._sigma_sq)**(1/7)

    def _compute_a(self):
        return ((945/4)*self._sigma_sq**3)**(1/7)

    def _sigma_sq_from_a(self, a):
        return (a / self._R) ** (self._alpha+1)

    def _sigma_sq_from_variance(self, variance):
        return (4/945)**(1/3) * (9*variance)**(7/6)

    def _support_size_from_sigma_sq(self, sigma_sq):
        a = ((945/4)*sigma_sq**3)**(1/7)
        return 2 * a

    def mean(self):
        return self._mu

    def variance(self):
        return ((-2*self._tau)/(1 + 2*self._alpha/(self._alpha-1)) *
                self._sigma_sq)

    def support_size(self):
        return 2*self._a

    def pdf(self, x):
        ndims = x.ndim - self._mu.ndim
        mu = self._mu[(None,)*ndims]
        sigma_sq = self._sigma_sq[(None,)*ndims]
        tau = self._tau[(None,)*ndims]
        return torch.clamp((1/3)*(-tau - .5*(x-mu)**2/sigma_sq), min=0)**3

    def sample(self, m):
        """Generate a random sample of size `m`."""
        raise NotImplementedError

    def tsallis_entropy(self):
        """Compute Tsallis alpha-entropy."""
        raise NotImplementedError


if __name__ == '__main__':
    from collections import OrderedDict
    from matplotlib import pyplot as plt
    import seaborn as sns

    text = "At least I was able to enjoy mocking the movie which is surprising since I was barely able to sit through it . In all honesty , my guess is the cover to the DVD case cost more than the entire movie . And saying that it is the same director as The Boogeyman , when a new version of that just came out ... nice touch guys , it was misleading enough to rope me in . The only thing that frustrated me more than the insufferable acting of the copycat was his haircut . Usually you only see that kind of hair on a ten year old boy and the character acted like it . The film looks like it was shot by a D + grad student of some film school excited to use every film technique he ever learned while attending classes .... sometimes , less is more buddy . Through out I would get lost by random plot twists that led nowhere or were unexplained . All this makes a bad movie but when the ending doesn ' t even come close to pulling it together , well , that makes it an exceptionally bad movie . Without a doubt this is the worst movie I have ever seen , and that includes my friends ' french final video for senior year of high school , but hey maybe i ' m a bit biased , I mean I did get to play an extra . P . S . I don ' t even think this deserves a star ... not even a half . NONE FOR YOU !!"
    words = text.split()
    length = len(words)

    so_mu, so_var = 0.5, 0.0044
    so_sigma_sq = 0.0044
    sp_mu, sp_var = 0.7914, 0.0001
    sp_sigma_sq = 2/3 * (5*sp_var)**(3/2)
    bi_mu, bi_var = 0.7102, 0.0025
    bi_sigma_sq = (1/15)**(1/2) * (7*bi_var)**(5/4)
    tri_mu, tri_var = 0.5763, 0.0277
    tri_sigma_sq = (1/15)**(1/2) * (7*tri_var)**(5/4)

    so = Gaussian1D(
        mu=torch.tensor([so_mu]),
        sigma_sq=torch.tensor([so_sigma_sq])
    )
    sp = SparsemaxGaussian1D(
        mu=torch.tensor([sp_mu]),
        sigma_sq=torch.tensor([sp_sigma_sq])
    )
    bi = BiweightGaussian1D(
        mu=torch.tensor([bi_mu]),
        sigma_sq=torch.tensor([bi_sigma_sq])
    )
    tri = TriweightGaussian1D(
        mu=torch.tensor([tri_mu]),
        sigma_sq=torch.tensor([tri_sigma_sq])
    )

    fig, ax = plt.subplots(figsize=(24, 3))
    x = torch.linspace(0, 1, length)

    entmaxes = OrderedDict({
        'softmax': so,
        'triweight': tri,
        'biweight': bi,
        'sparsemax': sp,
    })
    for label, entmax in entmaxes.items():
        y = entmax.pdf(x).numpy().flatten()
        if label != 'softmax':
            y[y == 0] = -1
        ax.plot(x.numpy(), y, '-', label=label)
        ax.fill_between(x, 0, y, alpha=.3)

    ax.set_xticks(x)
    ax.set_xticklabels(words, rotation=90, fontsize=6)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=1)

    # sns.despine()
    ax.legend()
    fig.tight_layout()
    plt.show()

