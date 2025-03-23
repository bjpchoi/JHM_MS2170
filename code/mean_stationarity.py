import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


petrarch_col1 = [160, 187, 187, 166, 193, 172, 190, 130, 163, 121, 145, 145]
boccaccio_df = pd.read_csv("Boccaccio.csv")
boccaccio_col = boccaccio_df.iloc[:, 1].dropna().values

dante_data = np.array([
    64, 58, 64, 49, 58, 85, 70, 70, 67, 64, 85, 61, 49, 58, 76, 64,
    64, 64, 67, 70, 61, 49, 52, 49, 49, 58, 64, 58, 61, 52, 55, 61, 43, 61
])

datasets = {
    "Dante": dante_data,
    "Petrarch": petrarch_col1,
    "Boccaccio": boccaccio_col
}

def mean_stationarity_test(data, n_iter=1000):
    """
    Test for mean stationarity by comparing the means of the first and second halves.

    Parameters:
        data: array-like time series.
        n_iter: number of random permutations.

    Returns:
        actual_diff: absolute difference between the means of the two halves for the actual data.
        perm_diffs: array of differences from permuted data.
        p_value: p-value for the test.
    """
    data = np.array(data)
    n = len(data)
    half = n // 2
    actual_diff = abs(np.mean(data[:half]) - np.mean(data[-half:]))

    perm_diffs = []
    for _ in range(n_iter):
        permuted = np.random.permutation(data)
        diff = abs(np.mean(permuted[:half]) - np.mean(permuted[-half:]))
        perm_diffs.append(diff)
    perm_diffs = np.array(perm_diffs)
    p_value = np.sum(perm_diffs >= actual_diff) / n_iter
    return actual_diff, perm_diffs, p_value


results = {}
n_iter = 1000

plt.figure(figsize=(15, 4))
for i, (label, data) in enumerate(datasets.items()):
    actual_diff, perm_diffs, p_value = mean_stationarity_test(data, n_iter=n_iter)
    results[label] = (actual_diff, p_value)

    plt.subplot(1, 3, i+1)
    plt.hist(perm_diffs, bins=30, alpha=0.75, color='gray', edgecolor='black')
    plt.axvline(actual_diff, color='red', linestyle='dashed', linewidth=2,
                label=f'Actual diff = {actual_diff:.2f}')
    plt.xlabel("Absolute Mean Difference")
    plt.ylabel("Frequency")
    plt.title(f"{label}\nMean Stationarity Test\np-value = {p_value:.3f}")
    plt.legend()

plt.tight_layout()
plt.show()

for label, (actual_diff, p_value) in results.items():
    print(f"{label}: Actual Mean Difference = {actual_diff:.2f}, p-value = {p_value:.3f}")
