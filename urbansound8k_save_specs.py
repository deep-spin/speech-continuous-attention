#!/usr/bin/python3
import os
import sys

import speechbrain as sb
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm.contrib import tqdm
from speechbrain import Stage
from speechbrain.utils.distributed import run_on_main

from continuous_attention import calculate_G, add_gaussian_basis_functions
from urbansound8k_train import dataio_prep
from urbansound8k_train import UrbanSound8kBrain as BaseUrbanSound8kBrain
from utils import configure_seed


class UrbanSound8kBrain(BaseUrbanSound8kBrain):

    def get_feats(self, batch, stage):
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)

        if self.hparams.amp_to_db:
            # try "magnitude" Vs "power"? db= 80, 50...
            Amp2db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
            feats = Amp2db(feats)

        # Normalization
        if self.hparams.normalize:
            feats = self.modules.mean_var_norm(feats, lens)

        # Recover lens
        batch_lens = (lens * feats.shape[1]).long()

        # Embeddings + sound classifier
        embeddings = self.modules.embedding_model(feats, lens)

        return wavs, lens, feats, embeddings

    def save_specs(
            self,
            test_set,
            max_key=None,
            min_key=None,
            progressbar=None,
            test_loader_kwargs={},
            num_samples=30,
            dname='specs/'
    ):
        from matplotlib import pyplot as plt
        test_loader_kwargs["batch_size"] = 1
        if not isinstance(test_set, torch.utils.data.DataLoader):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(test_set, Stage.TEST, **test_loader_kwargs)
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.modules.eval()
        n = 0
        if not os.path.exists(dname):
            os.makedirs(dname)
        with torch.no_grad():
            for batch in tqdm(test_set, dynamic_ncols=True, disable=not progressbar):
                wavs, lens, feats, embeddings = self.get_feats(batch, stage=Stage.TEST)
                batch_size, batch_len = wavs.shape
                ids = batch.id
                device = feats.device
                for i in range(batch_size):
                    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
                    spec_feats = feats[i].detach().cpu()
                    spec_conv = self.modules.embedding_model.conv_out[i]
                    spec_val_fn = self.modules.embedding_model.attn.val_fn_out[i]
                    axs = axes[0]
                    axs[0].set_title('Input', fontsize=11)
                    axs[0].imshow(spec_feats.t(), interpolation='nearest', origin='lower', aspect='auto')
                    axs[1].set_title('Conv out', fontsize=11)
                    axs[1].imshow(spec_conv.t(), interpolation='nearest', origin='lower', aspect='auto')
                    axs[2].set_title('Value fn p=0.1', fontsize=11)
                    axs[2].imshow(spec_val_fn.t(), interpolation='nearest', origin='lower', aspect='auto')

                    L = feats[i].shape[0]
                    values = self.modules.embedding_model.attn.values
                    nb_basis = 128
                    gaussian_sigmas = [0.03, 0.1, 0.3]
                    psi = [add_gaussian_basis_functions(nb_basis, sigmas=gaussian_sigmas, device=device)]
                    # psi = self.modules.embedding_model.attn.psis[L - 1]
                    spec_val_fn_2 = values.transpose(-1, -2).matmul(
                        calculate_G(psi, L, consider_pad=True, device=device, penalty=0.01)
                    )
                    spec_val_fn_3 = values.transpose(-1, -2).matmul(
                        calculate_G(psi, L, consider_pad=True, device=device, penalty=0.001)
                    )
                    spec_val_fn_4 = values.transpose(-1, -2).matmul(
                        calculate_G(psi, L, consider_pad=True, device=device, penalty=0.0001)
                    )
                    axs = axes[1]
                    axs[0].set_title('Value fn p=0.01', fontsize=11)
                    axs[0].imshow(spec_val_fn_2[i], interpolation='nearest', origin='lower', aspect='auto')
                    axs[1].set_title('Value out p=0.001', fontsize=11)
                    axs[1].imshow(spec_val_fn_3[i], interpolation='nearest', origin='lower', aspect='auto')
                    axs[2].set_title('Value fn p=0.0001', fontsize=11)
                    axs[2].imshow(spec_val_fn_4[i], interpolation='nearest', origin='lower', aspect='auto')

                    fname = os.path.join(dname, ids[i] + '.png')
                    fig.tight_layout()
                    plt.savefig(fname)
                    plt.close()
                    n += 1
                if n >= num_samples:
                    break


if __name__ == "__main__":

    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Configure seed for everything
    configure_seed(hparams["seed"])

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Tensorboard logging
    if hparams["use_tensorboard"]:
        from speechbrain.utils.train_logger import TensorboardLogger

        hparams["tensorboard_train_logger"] = TensorboardLogger(
            hparams["tensorboard_logs_folder"]
        )

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    datasets, label_encoder = dataio_prep(hparams)
    hparams["label_encoder"] = label_encoder

    class_labels = list(label_encoder.ind2lab.values())
    print("Class Labels:", class_labels)

    urban_sound_8k_brain = UrbanSound8kBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    urban_sound_8k_brain.save_specs(
        test_set=datasets["test"],
        min_key="error",
        progressbar=True,
        test_loader_kwargs=hparams["dataloader_options"],
        num_samples=hparams['attn_num_samples'],
        dname=hparams['spec_dname']
    )
