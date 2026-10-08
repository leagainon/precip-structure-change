from itertools import combinations

import numpy as np


# =============================
# 2D Variogram functions
# =============================
def compute_2d_variogram(gdf_ts, value_col="rre150h0", n_bins=4, max_lag=None):
    """
    Compute 2D experimental variogram surface for a single timestep.

    Parameters
    ----------
    gdf_ts   : **GeoDataFrame** with a geom in EPSG:3857
    value_col: str of the precipitation column (mm)
    n_bins   : int, number of bins per half-axis. The total grid is (2*n_bins+1)^2
    max_lag  : float, max lag in meters. Default: 1/2 of the max pairwise distance

    Returns
    -------
    gamma     : (2*n_bins+1, 2*n_bins+1) array, mean semivariance (mm^2)
    counts    : same shape, number of pairs per bin
    bin_edges : lag bin edges in meters
    """
    # drop nans
    gdf_ts = gdf_ts[gdf_ts[value_col].notna()]

    coords = np.array([[geom.x, geom.y] for geom in gdf_ts.geometry])
    values = gdf_ts[value_col].values

    pairs = list(
        combinations(range(len(gdf_ts)), 2)
    )  # all combinations of the indices of the gdf
    i_idx, j_idx = np.array(pairs).T  # fomatted in two arrays for better handling

    # Forward lags
    dx_fwd = coords[j_idx, 0] - coords[i_idx, 0]
    dy_fwd = coords[j_idx, 1] - coords[i_idx, 1]
    sv = 0.5 * (values[j_idx] - values[i_idx]) ** 2

    # Mirror lags: same semivariance since it is just the same lag but in the opposite direction
    dx = np.concatenate([dx_fwd, -dx_fwd])
    dy = np.concatenate([dy_fwd, -dy_fwd])
    sq_diff = np.concatenate([sv, sv])

    # Define the maxlag if not provided
    if max_lag is None:
        max_lag = 0.5 * np.sqrt(dx_fwd**2 + dy_fwd**2).max()

    # Define the edges
    bin_edges = np.linspace(-max_lag, max_lag, 2 * n_bins + 1)
    n_cells = 2 * n_bins

    # Compute the semivariances
    gamma = np.full((n_cells, n_cells), np.nan)
    counts = np.zeros((n_cells, n_cells), dtype=int)
    for bi in range(n_cells):
        for bj in range(n_cells):
            x0, x1 = bin_edges[bi], bin_edges[bi + 1]  # bin of the dx of the lag vector
            y0, y1 = bin_edges[bj], bin_edges[bj + 1]  # bin of the dy of the lag vector
            mask = (
                (dx >= x0) & (dx < x1) & (dy >= y0) & (dy < y1)
            )  # only keep the lag vectors that fall in the specified bin
            if mask.sum() > 0:
                gamma[bi, bj] = sq_diff[
                    mask
                ].mean()  # semivariance in this bin = mean square diff of the correspinding pair values
                counts[bi, bj] = (
                    mask.sum()
                )  # also compute the number of paris in this bin

    return gamma, counts, bin_edges
