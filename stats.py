import torch
from speechbrain.utils.metric_stats import MetricStats


class AttentionStats(MetricStats):
    def __init__(self, metric, n_jobs=1):
        super().__init__(metric, n_jobs=n_jobs)

    def append(self, ids, attn_stats):
        self.ids.extend(ids)
        self.scores.append(attn_stats)

    def summarize(self, field=None):
        if len(self.scores[0]) == 4:
            mus, vars, sigma_sqs, supps = zip(*self.scores)
        else:
            mus, vars, sigma_sqs, supps, _ = zip(*self.scores)
        self.summary = {
            "mu": torch.cat(mus).mean().item(),
            "var": torch.cat(vars).mean().item(),
            "sigma_sq": torch.cat(sigma_sqs).mean().item(),
            "supp": torch.cat(supps).mean().item(),
        }
        if field is not None:
            return self.summary[field]
        else:
            return self.summary

    def write_stats(self, filestream, verbose=False):
        if not self.summary:
            self.summarize()
        message = f"Average score: {self.summary['average']}\n"
        filestream.write(message)
        if verbose:
            print(message)
