#!/usr/bin/env python3
"""Helper to create Confusion Matrix figure
Authors
 * David Whipps 2021
 * Ala Eddine Limame 2021
"""
import random

import numpy as np
import matplotlib.pyplot as plt
import itertools

import torch
from torch.nn.utils.rnn import pad_sequence


def configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_cm_fig(cm, display_labels):

    fig = plt.figure(figsize=cm.shape, dpi=50, facecolor="w", edgecolor="k")
    ax = fig.add_subplot(1, 1, 1)

    ax.imshow(cm, cmap="Oranges")  # fits with the tensorboard colour scheme

    tick_marks = np.arange(cm.shape[0])

    ax.set_xlabel("Predicted class", fontsize=18)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(display_labels, ha="center", fontsize=18, rotation=90)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    ax.set_ylabel("True class", fontsize=18)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(display_labels, va="center", fontsize=18)
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()

    fmt = "d"  # TODO use '.3f' if normalized
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            verticalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=18,
        )

    fig.set_tight_layout(True)
    return fig


def plot_spec_with_attention(fname, feats, attn_kernel, attn_label):
    from collections import OrderedDict
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec

    length = feats.squeeze().shape[0]
    height = feats.squeeze().shape[1]

    fig = plt.figure(1, figsize=(9*2, 1.87*2))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.3, 0.7], width_ratios=[1])
    gs.update(
        left=0.05,
        right=0.95,
        bottom=0.08,
        top=0.93,
        wspace=0,
        hspace=0,
    )

    ax0 = plt.subplot(gs[0, 0])
    x = torch.arange(length+1)
    y = attn_kernel.pdf(x / length).numpy().flatten()
    if 'softmax' not in attn_label:
        y[y == 0] = -1
    ax0.plot(x.numpy(), y, '-', label=attn_label)
    ax0.fill_between(x, 0, y, alpha=.3)
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.axes.xaxis.set_visible(False)
    ax0.axes.yaxis.set_visible(False)
    ax0.set_ylim(bottom=0)
    ax0.set_xlim(left=x.min(), right=x.max())
    ax0.legend(ncol=1, shadow=False, frameon=False)

    ax1 = plt.subplot(gs[1, 0]) # place it where it should be.
    ax1.imshow(feats.squeeze().t(), interpolation='nearest', origin='lower')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Frequency')
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(left=x.min(), right=x.max())

    # fig.tight_layout()
    plt.show()
    plt.savefig(fname, dpi=300)


def detach_to_cpu(list_of_tensors):
    return [t.detach().cpu() for t in list_of_tensors]


def get_stats_continuous_attn(module):
    attn_stats = [module.mu, module.variance, module.sigma_sq, module.support_size]
    return detach_to_cpu(attn_stats)


def get_stats_discrete_attn(module, batch_lens):
    alphas = module.alphas
    pos = pad_sequence([torch.linspace(0, 1, n) for n in batch_lens.tolist()])
    pos = pos.t().unsqueeze(1).to(alphas.device)
    mu = (pos * alphas).sum(-1)
    variance = (pos.pow(2) * alphas).sum(-1) - mu.pow(2)
    sigma_sq = variance
    supp = (alphas > 0).float().mean(-1)
    attn_stats = [mu, variance, sigma_sq, supp, alphas.squeeze(1)]
    return detach_to_cpu(attn_stats)
