import mne
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np

from scipy.spatial import ConvexHull



sensors_bnci = [
        "Fz",
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "P1",
        "Pz",
        "P2",
        "POz",
    ]

sensors_dreyer = [
    'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'C1', 'C3', 'C5', 'C2', 'C4', 'C6', 'F4', 'FC2', 'FC4', 'FC6', 'CP2',
  'CP4', 'CP6', 'P4', 'F3', 'FC1', 'FC3', 'FC5', 'CP1', 'CP3', 'CP5', 'P3'
]
  
sensors = sensors_dreyer

montage = mne.channels.make_standard_montage('standard_1020')
all_positions = montage.get_positions()['ch_pos']

sensors_positions = np.array([all_positions[s] for s in sensors])

distances_to_center = np.linalg.norm(sensors_positions, axis=1)
indices_tries = np.argsort(distances_to_center)[::-1]
sensors_tries = [sensors[i] for i in indices_tries]

remaining_sensors = list(sensors)
groups = {}

for sensor in sensors_tries:
    if sensor not in remaining_sensors:
        continue

    dists = []
    for rem_s in remaining_sensors:
        d = np.linalg.norm(all_positions[sensor] - all_positions[rem_s])
        dists.append((rem_s, d))
    
    dists.sort(key=lambda x: x[1])

    current_group_names = [x[0] for x in dists[:3]]

    current_group_idx = [sensors.index(s) for s in current_group_names]
    groups[sensor] = current_group_names
    
    for member in current_group_names:
        remaining_sensors.remove(member)
        

# Création d'un montage EEG
montage = mne.channels.make_standard_montage('standard_1020')

info = mne.create_info(
    ch_names=sensors,
    sfreq=1000,
    ch_types='eeg'
)

info.set_montage(montage)

layout = mne.find_layout(info)

pos = layout.pos[:, :2]
names = layout.names

# Groupes de capteurs

groups = [groups[key] for key in groups]

matrix = np.load('./Results/GroupedPermutation/Groups_3/MDM/Dreyer2023C/With_cue/bad_subjects.npy')
feature_permutation = np.mean(matrix, axis=0)


# Figure
fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(pos[:, 0], pos[:, 1], c='k', s=60, zorder=5)

for x, y, name in zip(pos[:, 0], pos[:, 1], names):
    ax.text(x, y - 0.015, name,
            fontsize=8,
            ha='center',
            va='top',
            zorder=4)

# Dessin des frontières
cmap = plt.get_cmap('OrRd')
scores = [np.mean(feature_permutation[[sensors.index(ch) for ch in group if ch in sensors]]) for group in groups]
norm = Normalize(vmin=min(scores) if scores else 0, vmax=max(scores) if scores else 1)

for group, score in zip(groups, scores):

    idx = [names.index(ch) for ch in group if ch in names]

    color = cmap(norm(score))

    if len(idx) == 1:
        x, y = pos[idx[0]]
        ax.scatter([x], [y], s=260, facecolor=color, edgecolor='black', linewidth=1.2,
                   alpha=0.25, zorder=1)
        ax.text(x, y + 0.03, f"{score:.3f}",
                fontsize=9,
                ha='center',
                va='bottom',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'),
                zorder=4)
        continue

    if len(idx) == 2:
        points = pos[idx]
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=2.5, zorder=2)
        centroid = points.mean(axis=0)
        ax.text(centroid[0], centroid[1], f"{score:.3f}",
                fontsize=9,
                ha='center',
                va='center',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'),
                zorder=4)
        continue

    points = pos[idx]

    hull = ConvexHull(points)

    hull_points = points[hull.vertices]

    ax.fill(hull_points[:, 0], hull_points[:, 1],
            color=color, alpha=0.25, zorder=1)
    
    hull_points_closed = np.vstack([hull_points, hull_points[0]])
    ax.plot(hull_points_closed[:, 0], hull_points_closed[:, 1],
            color=color, linewidth=2.5, zorder=2)
    
    centroid = hull_points.mean(axis=0)
    ax.text(centroid[0], centroid[1], f"{score:.3f}",
            fontsize=9,
            ha='center',
            va='center',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'),
            zorder=4)

# Add colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, label='Permutation Score')


ax.set_title("Clusters de capteurs EEG")
ax.set_aspect('equal')
ax.axis('off')
plt.savefig("Results/GroupedPermutation/Groups_3/MDM/Dreyer2023C/With_cue/Cluster_Capteurs_bad_sujets.png", dpi=300)
plt.show()