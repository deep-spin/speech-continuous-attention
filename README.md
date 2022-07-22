# speech-continuous-attention

Speech Classification with Continuous Attention Mechanisms.

This is the code for speech classification experiments of the paper:
- [Sparse Continuous Distributions and Fenchel-Young Losses](https://arxiv.org/abs/2108.01988) by André F. T. Martins, Marcos Treviso, António Farinhas, Pedro M. Q. Aguiar, Mário A. T. Figueiredo, Mathieu Blondel, Vlad Niculae.

which builds upon:
- [Sparse and Continuous Attention Mechanisms](https://papers.neurips.cc/paper/2020/hash/f0b76267fbe12b936bd65e203dc675c1-Abstract.html) by André Martins, António Farinhas, Marcos Treviso, Vlad Niculae, Pedro Aguiar, Mario Figueiredo. NeurIPS 2020.


The code is based on the following `speechbrain` recipes:
- [UrbanSound8k](https://github.com/speechbrain/speechbrain/tree/develop/recipes/UrbanSound8k)
- [VoxCeleb](https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxCeleb/SpeakerRec)

We provide scripts for UrbanSound8k only. The datasets should be placed in the `data/` folder.


## Installation

First, install the [spcdist](https://github.com/deep-spin/sparse_continuous_distributions) library to get support for sparse continuous distributions:
```
pip3 install git+https://github.com/deep-spin/sparse_continuous_distributions#egg=spcdist
```

Then install other requirements via:
```
pip3 install -r requirements.txt
```


### Run 10-fold

Valdiating and testing on fold 1:
```sh
# discrete-attention models
for max_act in "softmax", "entmax1333", "entmax15", "sparsemax"
do
    mkdir "results/urbansound8k_discrete_${max_act}"
    python3 urbansound8k_train.py hparams/urbansound8k_acrnn.yaml \
        --train_fold_nums=[2, 3, 4, 5, 6, 7, 8, 9, 10] \
        --valid_fold_nums=[1] \
        --test_fold_nums=[1] \
        --attn_domain discrete \
        --attn_max_activation ${max_act} \
        --output_folder=results/urbansound8k_discrete_${max_act}/fold_1 \
        --device=cuda:0
done

# continuous-attention models
for max_act in "softmax", "triweight", "biweight", "sparsemax"
do
    mkdir "results/urbansound8k_continuous_${max_act}"
    python3 urbansound8k_train.py hparams/urbansound8k_acrnn.yaml \
        --train_fold_nums=[2, 3, 4, 5, 6, 7, 8, 9, 10] \
        --valid_fold_nums=[1] \
        --test_fold_nums=[1] \
        --attn_domain continuous \
        --attn_max_activation ${max_act} \
        --output_folder=results/urbansound8k_continuous_${max_act}/fold_1 \
        --device=cuda:0 
done
```

Do the same for the other folds and average in the end to get the final results as in the paper. 
See the config files in the `hparams` folder for more information.


### Save and plot attention maps

First, save the spectrograms:
```sh
mkdir -p specs/urbansound8k_continuous_sparsemax
python3 urbansound8k_save_specs.py hparams/urbansound8k_acrnn.yaml \
    --train_fold_nums=[2, 3, 4, 5, 6, 7, 8, 9, 10] \
    --valid_fold_nums=[1] \
    --test_fold_nums=[1] \
    --attn_domain continuous \
    --attn_max_activation sparsemax \
    --output_folder=results/urbansound8k_continuous_sparsemax/fold_1 \
    --device=cpu \
    --attn_num_samples 50 \
    --spec_dname specs/urbansound8k_continuous_sparsemax/
```

And then save the attention maps:
```sh
mkdir saved_attentions
python3 urbansound8k_save_attentions.py hparams/urbansound8k_acrnn.yaml \
    --train_fold_nums=[2, 3, 4, 5, 6, 7, 8, 9, 10] \
    --valid_fold_nums=[1] \
    --test_fold_nums=[1] \
    --attn_domain continuous \
    --attn_max_activation sparsemax \
    --output_folder=results/urbansound8k_continuous_sparsemax/fold_1 \
    --device=cuda:2 \
    --attn_num_samples 200 \
    --attn_fname saved_attentions/urbansound8k_continuous_sparsemax.csv
```

For plotting attention densities you can follow the step-by-step instructions in this notebook: https://colab.research.google.com/drive/1Ce51VB_rgmNmxB5lggRSUDZ__-zu-Xov?usp=sharing


### Citation
If you use this codebase, or otherwise found our work valuable, please cite:
```
todo
```

