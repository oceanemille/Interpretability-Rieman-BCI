import numpy as np
from scipy.stats import spearmanr, weightedtau, kendalltau


def _ranking_scores(importance, rank_by):
    values = np.asarray(importance, dtype=float)
    if values.ndim != 1:
        raise ValueError(
            f"importance must have shape (n_channels,), got {values.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("importance must contain only finite values.")
    if rank_by == "value":
        return values
    if rank_by == "magnitude":
        return np.abs(values)
    raise ValueError("rank_by must be either 'value' or 'magnitude'.")


def importance_to_ranking(importance, channel_names, rank_by="value"):
    """Return channel names ordered from most to least important."""
    scores = _ranking_scores(importance, rank_by)
    channel_names = np.asarray(channel_names)
    if channel_names.ndim != 1 or len(channel_names) != len(scores):
        raise ValueError(
            "channel_names must be one-dimensional and match the number "
            "of importance values."
        )
    if len(np.unique(channel_names)) != len(channel_names):
        raise ValueError("channel_names must be unique.")

    order = np.argsort(-scores, kind="stable")
    return channel_names[order].tolist()


def top_k_overlap(importance_a, importance_b, k=5, rank_by="value"):
    """Return the proportion of shared channels in two top-k rankings."""
    scores_a = _ranking_scores(importance_a, rank_by)
    scores_b = _ranking_scores(importance_b, rank_by)
    if scores_a.shape != scores_b.shape:
        raise ValueError("The two importance arrays must have the same shape.")
    if not 1 <= k <= len(scores_a):
        raise ValueError(f"k must be between 1 and {len(scores_a)}, got {k}.")

    top_a = set(np.argsort(-scores_a, kind="stable")[:k])
    top_b = set(np.argsort(-scores_b, kind="stable")[:k])
    return len(top_a & top_b) / k


def compare_rankings(
    importance_a,
    importance_b,
    channel_names,
    k=5,
    rank_by="value",
):
    """Compare two channel-importance rankings with three agreement metrics.

    ``rank_by='value'`` preserves the direction of signed attributions, while
    ``rank_by='magnitude'`` compares their absolute strength.
    """
    scores_a = _ranking_scores(importance_a, rank_by)
    scores_b = _ranking_scores(importance_b, rank_by)
    if scores_a.shape != scores_b.shape:
        raise ValueError("The two importance arrays must have the same shape.")

    ranking_a = importance_to_ranking(scores_a, channel_names)
    ranking_b = importance_to_ranking(scores_b, channel_names)
    spearman = spearmanr(scores_a, scores_b).statistic
    kendall_tau,_ = kendalltau(scores_a, scores_b)
    weighted_kendall_tau,_ = weightedtau(scores_a,  scores_b)

    return {
        "spearman": float(spearman),
        "top_k_overlap": top_k_overlap(scores_a, scores_b, k=k),
        "kendall_tau" : kendall_tau,
        "weighted_kendall_tau":weighted_kendall_tau
    }
