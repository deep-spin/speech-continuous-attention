#!/usr/bin/python3
import sys

import speechbrain as sb
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from tqdm.contrib import tqdm
from speechbrain import Stage
from speechbrain.utils.distributed import run_on_main

from urbansound8k_train import dataio_prep
from urbansound8k_train import UrbanSound8kBrain as BaseUrbanSound8kBrain
from utils import configure_seed, get_stats_continuous_attn, get_stats_discrete_attn


class UrbanSound8kBrain(BaseUrbanSound8kBrain):

    def get_attention(self, batch, stage):
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
        logprobas = self.modules.classifier(embeddings)
        predictions = logprobas.cpu().detach().argmax(-1).squeeze(-1)

        # Get attention
        if hasattr(self.modules.embedding_model, 'asp'):
            if hasattr(self.modules.embedding_model.asp, 'cont_attn'):
                attn_stats = get_stats_continuous_attn(self.modules.embedding_model.asp.cont_attn)
            else:
                attn_stats = get_stats_discrete_attn(self.modules.embedding_model.asp, batch_lens)
        else:
            if hasattr(self.modules.embedding_model.attn, 'cont_max_activation'):  # continuous case
                attn_stats = get_stats_continuous_attn(self.modules.embedding_model.attn)
            else:
                attn_stats = get_stats_discrete_attn(self.modules.embedding_model.attn, batch_lens)

        return wavs, lens, feats, attn_stats, predictions

    def save_attentions(
            self,
            test_set,
            max_key=None,
            min_key=None,
            progressbar=None,
            test_loader_kwargs={},
            num_samples=30,
            fname='attns.txt'
    ):
        import pandas as pd
        test_loader_kwargs["batch_size"] = 1
        if not isinstance(test_set, torch.utils.data.DataLoader):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(test_set, Stage.TEST, **test_loader_kwargs)
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.modules.eval()
        n = 0
        df = pd.DataFrame(
            columns=['file_id', 'label', 'pred', 'duration', 'fold', 'mu', 'var', 'sigma_sq', 'supp', 'alpha']
        )
        with torch.no_grad():
            for batch in tqdm(test_set, dynamic_ncols=True, disable=not progressbar):
                wavs, lens, feats, attn_stats, preds = self.get_attention(batch, stage=Stage.TEST)
                batch_size, batch_len = wavs.shape
                ids = batch.id
                labels, _ = batch.class_string_encoded
                durs = (lens * batch_len).long()
                if len(attn_stats) == 5:
                    mus, vars, sigma_sqs, supps, alphas = attn_stats
                else:
                    alphas = None
                    mus, vars, sigma_sqs, supps = attn_stats
                for i in range(batch_size):
                    alpha = ';'.join(['{:.4f}'.format(a) for a in alphas[i].tolist()]) if alphas is not None else None
                    df = df.append({
                        'file_id': ids[i],
                        'label': self.hparams.label_encoder.ind2lab[labels[i].item()],
                        'pred': self.hparams.label_encoder.ind2lab[int(preds[i].item())],
                        'duration': durs[i].item(),
                        'fold': self.hparams.test_fold_nums[0],
                        'mu': mus[i].item(),
                        'var': vars[i].item(),
                        'sigma_sq': sigma_sqs[i].item(),
                        'supp': supps[i].item(),
                        'alpha': alpha
                    }, ignore_index=True)
                    n += 1
                if n >= num_samples:
                    break
        df.to_csv(fname)


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

    # Load the best checkpoint for saving attention
    urban_sound_8k_brain.save_attentions(
        test_set=datasets["test"],
        min_key="error",
        progressbar=True,
        test_loader_kwargs=hparams["dataloader_options"],
        num_samples=hparams['attn_num_samples'],
        fname=hparams['attn_fname']
    )
