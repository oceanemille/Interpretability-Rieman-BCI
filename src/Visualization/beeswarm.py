
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def shap_beeswarm(
    shap_values,
    sensor_names,
    subject=None,
    out_dir=None,
    title=None,
    max_channels=None,
    ax=None,
    show=True,
):
    """Plot trial-level Shapley distributions for each EEG channel.

    Parameters
    ----------
    shap_values : array-like, shape (n_trials, n_channels)
    sensor_names : sequence of str, length n_channels
    max_channels : int or None
        If set, display only the channels with the largest mean absolute
        Shapley value.

    Returns
    -------
    fig, ax
        Matplotlib objects that can be customized by the caller.
    """
    shap_values = np.asarray(shap_values)
    sensor_names = np.asarray(sensor_names)
    if shap_values.ndim != 2:
        raise ValueError(
            "shap_values must have shape (n_trials, n_channels), "
            f"got {shap_values.shape}."
        )
    if sensor_names.ndim != 1 or len(sensor_names) != shap_values.shape[1]:
        raise ValueError(
            "sensor_names must be one-dimensional and match n_channels."
        )

    channel_order = np.argsort(-np.mean(np.abs(shap_values), axis=0))
    if max_channels is not None:
        if not 1 <= max_channels <= len(sensor_names):
            raise ValueError(
                f"max_channels must be between 1 and {len(sensor_names)}."
            )
        channel_order = channel_order[:max_channels]

    selected_names = sensor_names[channel_order]
    selected_values = shap_values[:, channel_order]
    df = pd.DataFrame({
        "channel": np.tile(selected_names, selected_values.shape[0]),
        "shap": selected_values.reshape(-1),
    })

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    sns.stripplot(
        data=df,
        x="shap",
        y="channel",
        order=selected_names,
        jitter=0.3,
        alpha=0.7,
        ax=ax,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Shapley value for class left_hand")
    ax.set_ylabel("Channel")
    ax.set_title(title or f"Trial-level Shapley values{f' — {subject}' if subject else ''}")
    fig.tight_layout()

    if out_dir is not None:
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{subject}" if subject is not None else ""
        fig.savefig(output_dir / f"shapley_beeswarm{suffix}.pdf", dpi=300)
    if show:
        plt.show()
    return fig, ax

def filter_regions(regions, sensor_names):
    sensor_set = set(sensor_names)
    filtered = {}

    for region, chans in regions.items():
        valid = [ch for ch in chans if ch in sensor_set]
        if len(valid) > 0:
            filtered[region] = valid

    return filtered

def shap_beeswarm_grouped(
    shap_values,
    sensor_names,
    dataset,
    subject,
    regions,
    channel_to_hemisphere,
    out_dir,
    title=None
):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    n_trials, n_channels = shap_values.shape

    # --- dataframe ---
    data = []
    for ch in range(n_channels):
        for i in range(n_trials):
            data.append({
                "channel": sensor_names[ch],
                "shap": shap_values[i, ch]
            })

    df = pd.DataFrame(data)

    # --- region filtering ---
    regions_used = filter_regions(regions, sensor_names)

    channel_to_region = {}
    for region, chans in regions_used.items():
        for ch in chans:
            channel_to_region[ch] = region

    df["region"] = df["channel"].map(channel_to_region)
    df["hemisphere"] = df["channel"].map(channel_to_hemisphere)

    # remove unmapped
    df = df.dropna(subset=["region", "hemisphere"])

    # order regions
    region_order = list(regions_used.keys())
    df["region"] = pd.Categorical(df["region"], categories=region_order, ordered=True)

    # --- plot ---
    plt.figure(figsize=(12, 6))

    ax = sns.stripplot(
        data=df,
        x="shap",
        y="region",
        hue="hemisphere",  
        jitter=0.25,
        alpha=0.7,
        size=5
    )

    ax.set_xlabel("Shapley values for class left_hand")
    ax.set_ylabel("Regions")


    plt.axvline(0, color="black", linewidth=1)

    # separation lines between regions
    for i in range(len(region_order) - 1):
        plt.axhline(i + 0.5, color="grey", linewidth=0.6, alpha=0.5)

    plt.title(title or f"SHAP grouped regions dataset {dataset} - subject {subject}")

    plt.legend(title="Hemisphere", loc="upper right")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/Shap_beeswarm_grouped_lat_{subject}.pdf", dpi=300)
    plt.show()
