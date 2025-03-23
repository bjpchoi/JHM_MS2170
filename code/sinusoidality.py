import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit, minimize_scalar
from scipy.stats import f

petrarch_col1 = [160, 187, 187, 166, 193, 172, 190, 130, 163, 121, 145, 145]

boccaccio_df = pd.read_csv("Boccaccio.csv")
boccaccio_col = boccaccio_df.iloc[:, 1].dropna().values

dante_data = np.array([
    64, 58, 64, 49, 58, 85, 70, 70, 67, 64, 85, 61, 49, 58, 76, 64,
    64, 64, 67, 70, 61, 49, 52, 49, 49, 58, 64, 58, 61, 52, 55, 61, 43, 61
])


def sinusoid(t, A, omega, phi, C):
    """Sinusoidal model: A*sin(omega*t + phi) + C"""
    return A * np.sin(omega * t + phi) + C


def fit_sinusoid(data, label, window=1):
    """
    Fit a sinusoidal model to mean-subtracted, smoothed data using standard
    nonlinear least squares (vertical residuals). Then, compute both the normal (vertical)
    R² and an extra orthogonal R² computed from perpendicular distances to the fitted curve.

    Parameters:
        data   : array-like raw data.
        label  : string identifier.
        window : window size for moving average smoothing.

    Returns:
        t_smoothed: time indices for the smoothed data.
        y         : mean-subtracted, smoothed data.
        fitted    : fitted sinusoidal model values.
        params    : fitted parameters [A, omega, phi, C].
        r2_normal : R² computed from vertical residuals.
        r2_orth   : R² computed from orthogonal (perpendicular) residuals.
    """
    data = np.array(data)
    smoothed_data = np.convolve(data, np.ones(window) / window, mode='valid')
    t_smoothed = np.arange(len(smoothed_data)) + (window - 1) / 2.0

    y = smoothed_data - np.mean(smoothed_data)

    A0 = (np.max(y) - np.min(y)) / 2.0
    omega0 = 2 * np.pi / (len(y) / 2) 
    initial_guess = [A0, omega0, 0, 0]

    # increased maxfev to avoid runtime errors during bootstrap simulations.
    params, _ = curve_fit(sinusoid, t_smoothed, y, p0=initial_guess, maxfev=10000)
    fitted = sinusoid(t_smoothed, *params)

    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_normal = 1 - ss_res / ss_tot

    def ortho_dist_sq(t_i, y_i, params):
        A, omega, phi, C = params
        period = 2 * np.pi / abs(omega) if omega != 0 else 10.0
        bounds = (t_i - period/2, t_i + period/2)
        res = minimize_scalar(lambda tau: (tau - t_i)**2 + (sinusoid(tau, A, omega, phi, C) - y_i)**2,
                              bounds=bounds, method='bounded')
        return res.fun

    ss_res_orth = np.sum([ortho_dist_sq(t, yi, params) for t, yi in zip(t_smoothed, y)])
    centroid_t = np.mean(t_smoothed)
    centroid_y = np.mean(y)
    ss_tot_orth = np.sum((t_smoothed - centroid_t)**2 + (y - centroid_y)**2)
    r2_orth = 1 - ss_res_orth / ss_tot_orth

    return t_smoothed, y, fitted, params, r2_normal, r2_orth


def compute_p_value(r2, n, p=4):
    """
    Compute the F-statistic and corresponding p-value for a model given its R²,
    number of data points n, and number of model parameters p.

    Parameters:
        r2 : coefficient of determination.
        n  : number of data points.
        p  : number of parameters in the model (default is 4).

    Returns:
        F_stat  : F-statistic.
        p_value : p-value from the F-distribution.
    """
    dfn = p - 1     
    dfd = n - p     
    F_stat = (r2 / dfn) / ((1 - r2) / dfd)
    p_value = f.sf(F_stat, dfn, dfd)
    return F_stat, p_value


fig = plt.figure(figsize=(16, 8))
outer_gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 2], wspace=0.2)

left_gs = outer_gs[0].subgridspec(2, 1, hspace=0.3)
ax_left_top = fig.add_subplot(left_gs[0])
x1 = np.arange(len(petrarch_col1))
ax_left_top.plot(
    x1, petrarch_col1,
    marker='_',
    markersize=12,
    color='blue',
    markeredgewidth=3
)
ax_left_top.set_title('Petrarch, I Trionfi (by chapter)')
ax_left_top.set_xlabel('Index')
ax_left_top.set_ylabel('Length')

ax_left_bottom = fig.add_subplot(left_gs[1])
x3 = np.arange(len(boccaccio_col))
ax_left_bottom.plot(
    x3, boccaccio_col,
    marker='_',
    markersize=6,
    color='red',
    markeredgewidth=3
)
ax_left_bottom.set_title('Boccaccio, Decameron')
ax_left_bottom.set_xlabel('Index')
ax_left_bottom.set_ylabel('Length')

right_gs = outer_gs[1].subgridspec(3, 1, hspace=0.5)

datasets = {
    "Dante": dante_data,
    "Petrarch": petrarch_col1,
    "Boccaccio": boccaccio_col
}

results = {}

for i, (label, data) in enumerate(datasets.items()):
    t_smoothed, y, fitted, params, r2_norm, r2_orth = fit_sinusoid(data, label, window=1)
    vertical_corr = np.sqrt(r2_norm) if r2_norm >= 0 else np.nan
    orthogonal_corr = np.sqrt(r2_orth) if r2_orth >= 0 else np.nan
    results[label] = (params, vertical_corr, orthogonal_corr)

    ax_right = fig.add_subplot(right_gs[i])
    ax_right.plot(t_smoothed, y, 'o', label=f'{label}')
    ax_right.plot(t_smoothed, fitted, '-', label='Sinusoidal fit')
    ax_right.set_title(f'{label} Sinusoidality: Correlation (orthogonal R) = {orthogonal_corr:.3f}')
    ax_right.set_xlabel('Index')
    ax_right.set_ylabel('Length (Centered)')
    ax_right.legend()

plt.tight_layout()
plt.show()
