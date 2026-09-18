import mne
import matplotlib.pyplot as plt
from mne.viz import plot_topomap as plot_topomap_mne
import math
import numpy as np

def plot_topomap(
    importance_values,
    sensors,
    title,
    vlim=None,
    ax=None,
    savefile_name=None,
    montage_type="standard_1020",
    show_cbar=True,
    cbar_type = "Shapley",
    cmap='RdBu_r',
    cbar_title = 'Importance',
    suptitle = "Shapley_values on dataset"
):

    montage = mne.channels.make_standard_montage(montage_type)
    info = mne.create_info(ch_names=sensors, sfreq=500, ch_types="eeg")
    info.set_montage(montage)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        standalone = True
    else:
        fig = ax.get_figure() 
        standalone = False
    
    if vlim is None:
        vlim = (np.max(np.abs(importance_values)), -np.max(np.abs(importance_values)))

    if cbar_type == 'Shapley' : 
        cmap = 'PiYG'

    im, _ = plot_topomap_mne(
        importance_values,
        info,
        axes=ax,
        show=False,
        cmap=cmap,
        extrapolate = 'local',
        vlim=vlim
    )

    ax.set_title(title)

    if show_cbar:
        if cbar_type == 'Shapley':
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            #cbar.set_label(cbar_title)

            cbar.ax.text(
                0.5, 0.98,
                "Left Hand",
                ha="center",
                va="top",
                fontsize=8,
                transform=cbar.ax.transAxes
            )

            cbar.ax.text(
                0.5, 0.02,
                "Right Hand",
                ha="center",
                va="bottom",
                fontsize=8,
                transform=cbar.ax.transAxes
            )

        else :
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.08)
            cbar.set_label(cbar_title)


    if standalone:
        #plt.tight_layout()
        plt.savefig(savefile_name, bbox_inches='tight')
        plt.show()
    
    else :
        plt.suptitle(suptitle, fontsize=16)
        
        
        
def plot_pannel(values, 
subject_names, 
sensors, 
scores, 
savefile_path="/home/omille/Documents/Stage/Rendu/Panel_Tous_Sujets.pdf", 
cbar_type = 'Shapley', 
suptitle = "Shapley values on dataset"):
    n_subjects = len(values)
    n_cols = 4
    n_rows = math.ceil(n_subjects / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()
    v_max = np.max(np.abs(values))
    v_min = -v_max

    for i in range(n_subjects):
        subject_values = values[i]
        score = scores[i]
        
        plot_topomap(
            importance_values=subject_values, 
            sensors=sensors,
            title=f"{subject_names[i]}\nMean Score {score:.2f} ",
            ax=axes[i],
            show_cbar=True,
            cbar_type = cbar_type,
            vlim=(v_min,v_max),
            suptitle = suptitle 
        )

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.savefig(savefile_path, bbox_inches='tight')
    plt.show()

