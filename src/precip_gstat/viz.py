import matplotlib.pyplot as plt
import numpy as np


def plot_2d_variogram(
    gamma, counts, bin_edges, title="2D Variogram", min_pairs=2, fs=16
):
    plt.rcParams.update({"font.size": 16})

    gamma_masked = np.where(counts >= min_pairs, gamma, np.nan)

    # Use edges directly with shading="flat" so that pcolormesh expects coordinate arrays of length n+1 for an (n,n) data array
    edges_km = bin_edges / 1000  # m to km, length = 2*n_bins+1

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    im0 = axes[0].pcolormesh(
        edges_km, edges_km, gamma_masked.T, cmap="RdYlGn_r", shading="flat"
    )
    plt.colorbar(im0, ax=axes[0], label=r"Semivariance (mm$^2$)")
    axes[0].set_xlabel("Lag Easting (km)")
    axes[0].set_ylabel("Lag Northing (km)")
    axes[0].set_title(title, fontsize=fs)
    axes[0].axhline(0, color="k", lw=0.5, ls="--")
    axes[0].axvline(0, color="k", lw=0.5, ls="--")
    axes[0].set_aspect("equal")

    im1 = axes[1].pcolormesh(edges_km, edges_km, counts.T, cmap="Blues", shading="flat")
    plt.colorbar(im1, ax=axes[1], label="Pair count")
    axes[1].set_xlabel("Lag Easting (km)")
    axes[1].set_ylabel("Lag Northing (km)")
    axes[1].set_title("pair counts")
    axes[1].axhline(0, color="k", lw=0.5, ls="--")
    axes[1].axvline(0, color="k", lw=0.5, ls="--")
    axes[1].set_aspect("equal")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig
