
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def shap_beeswarm(shap_values, sensor_names, subject,OUT_DIR):
    # X_diff_power : (n_trials, n_channels)

    n_trials, n_channels = shap_values.shape
    data = []
    for ch in range(n_channels):
        for i in range(n_trials):
            data.append({
                "channel": sensor_names[ch],
                "shap": shap_values[i, ch],
                #"log_power_diff": X_diff_power[i, ch]
            })
    df = pd.DataFrame(data)

    
    plt.figure(figsize=(12, 6))
    scatter = sns.stripplot(
        data=df,
        x="shap",
        y="channel",
        #hue="log_power_diff",
        #palette="coolwarm",
        jitter=0.3,
        alpha=0.7,
        legend=False
    )
    
    # Colorbar
    #norm = plt.Normalize(df["log_power_diff"].min(), df["log_power_diff"].max())
    #sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    #plt.colorbar(sm, ax=scatter.axes, label="Log-power vs baseline")
    
    plt.axvline(0, color="black", linewidth=1)
    plt.savefig(f'{OUT_DIR}/Shapley_plot_sujet_{subject}.pdf')
    plt.show()

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