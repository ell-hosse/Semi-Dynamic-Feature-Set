import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def plot_features(expanded_set, static_features_size, dynamic_features_size, static_features_name_list=None):
    try:
        import torch
        if isinstance(expanded_set, torch.Tensor):
            expanded_set = expanded_set.detach().cpu().numpy()
    except ImportError:
        pass

    arr = expanded_set
    if arr.ndim == 3:
        arr = arr[0]
    elif arr.ndim != 2:
        raise ValueError(f"expanded_set must be (N, W, F) or (W, F), got shape {arr.shape}")

    W, F = arr.shape
    if static_features_size + dynamic_features_size != F:
        raise ValueError(
            f"static_features_size + dynamic_features_size ({static_features_size}+{dynamic_features_size}) "
            f"must equal F ({F})"
        )

    static_part  = arr[:, :static_features_size] # (W, F_s)
    dynamic_part = arr[:, static_features_size: F]  # (W, F_d)

    fig, ax = plt.subplots(figsize=(12, 5))
    t = np.arange(W)

    static_labels = []
    for i in range(static_features_size):
        lbl = None
        if static_features_name_list and len(static_features_name_list) == static_features_size:
            lbl = static_features_name_list[i]
            static_labels.append(lbl)
        else:
            lbl = f"static_feature_{i+1}"
            static_labels.append(lbl)

        ax.plot(t, static_part[:, i], color="black", linewidth=1.2, alpha=0.85, label=lbl if i == 0 else None)
        if static_features_name_list and len(static_features_name_list) == static_features_size:
            ax.annotate(static_features_name_list[i],
                        xy=(W-1, static_part[-1, i]),
                        xytext=(5, 0),
                        textcoords="offset points",
                        fontsize=9,
                        color="black",
                        va="center")

    if dynamic_features_size > 0:
        cmap = get_cmap("tab20") if dynamic_features_size > 10 else get_cmap("tab10")
        for j in range(dynamic_features_size):
            color = cmap(j % cmap.N)
            ax.plot(t, dynamic_part[:, j],
                    linewidth=1.6,
                    label=f"dynamic_feature_{j+1}",
                    color=color)

    ax.set_title("Static (black) and Dynamic (colored) Features over Window")
    ax.set_xlabel("Time (within window)")
    ax.set_ylabel("Feature value")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=9, frameon=True)

    plt.tight_layout()
    plt.show()
