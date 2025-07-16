from scipy.stats import ttest_rel, pearsonr, spearmanr
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from dtaidistance import dtw
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

results_list = []

for file in tqdm(os.listdir(os.path.join('output', 'csv'))):
    data = pd.read_excel(os.path.join('output', 'csv', file))
    data.sort_values(by=['folder', 'filename'], inplace=True)
    data = data[data['filtered_iri'] != 0]
    filtered = data['filtered_iri'].to_numpy()
    gt = data['gt_mesh'].to_numpy()

    if len(filtered) == 0 or len(gt) == 0:
        continue

    diff = filtered - gt

    # DTW Distance
    dtw_distance = dtw.distance(filtered, gt)

    # Area Between Curves (ABC)
    abc = np.trapz(np.abs(diff))

    # Cross-Correlation Lag
    corr = np.correlate(filtered - np.mean(filtered),
                        gt - np.mean(gt), mode='full')
    lag = np.argmax(corr) - (len(filtered) - 1)

    # Peak counts
    peaks_filtered, _ = find_peaks(filtered)
    peaks_gt, _ = find_peaks(gt)
    peak_count_diff = abs(len(peaks_filtered) - len(peaks_gt))
    fdtw, path = fastdtw(filtered, gt, dist=2)

    results_list.append({
        'filename': file.split('.')[0],
        'rmse': np.sqrt((diff ** 2).mean()),
        'mae': np.mean(np.abs(diff)),
        't_stat': ttest_rel(filtered, gt)[0],
        'p_value': ttest_rel(filtered, gt)[1],
        'dtw': dtw_distance,
        'abc': abc,
    })

# Save results
results = pd.DataFrame(results_list)
results.sort_values(by='filename', inplace=True)
results.to_excel(os.path.join('output', 'results_all.xlsx'), index=False)
