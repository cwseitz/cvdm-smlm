import numpy as np
import matplotlib.pyplot as plt
import os

BASE_PATH = '/Users/seitzcx/Desktop/exps/exp_CVDM/results'

prefixes_100 = [
'cvdm_sz64_n1000_N100-101_0bb3fc88',
'cvdm_sz64_n1000_N100-101_2cefe591',
'cvdm_sz64_n1000_N100-101_6775a814',
'cvdm_sz64_n1000_N100-101_6fea8ebc',
'cvdm_sz64_n1000_N100-101_8ba68f8c',
'cvdm_sz64_n1000_N100-101_a2900756',
'cvdm_sz64_n1000_N100-101_c3b595da',
'cvdm_sz64_n1000_N100-101_d983656b',
'cvdm_sz64_n1000_N100-101_e1a90762'
]

prefixes_200 = [
'cvdm_sz64_n1000_N200-201_08e0cb8f',
'cvdm_sz64_n1000_N200-201_1064a130',
'cvdm_sz64_n1000_N200-201_17711672',
'cvdm_sz64_n1000_N200-201_4b994678',
'cvdm_sz64_n1000_N200-201_5e863f28',
'cvdm_sz64_n1000_N200-201_6345e77e',
'cvdm_sz64_n1000_N200-201_73d97b21',
'cvdm_sz64_n1000_N200-201_9b63ec82',
'cvdm_sz64_n1000_N200-201_ddafbc59',
'cvdm_sz64_n1000_N200-201_f27efd73'
]
prefixes_500 = [
'cvdm_sz64_n1000_N500-501_08579b5a',
'cvdm_sz64_n1000_N500-501_1ca31d33',
'cvdm_sz64_n1000_N500-501_3277fdae',
'cvdm_sz64_n1000_N500-501_422ca18a',
'cvdm_sz64_n1000_N500-501_4e13f5e4',
'cvdm_sz64_n1000_N500-501_6b94f127',
'cvdm_sz64_n1000_N500-501_7120a4e5',
'cvdm_sz64_n1000_N500-501_867a95c3',
'cvdm_sz64_n1000_N500-501_c9acfc1c',
'cvdm_sz64_n1000_N500-501_e1305d0b'
]


def load_metrics(prefixes):
    metrics_list = []
    for prefix in prefixes:
        npz_path = os.path.join(BASE_PATH, f"{prefix}-set.npz")
        metrics = np.load(npz_path, allow_pickle=True)['metrics']
        metrics = np.vstack(metrics)
        metrics_list.append(metrics)
    return np.vstack(metrics_list)

metrics_100 = load_metrics(prefixes_100)
metrics_200 = load_metrics(prefixes_200)
metrics_500 = load_metrics(prefixes_500)

def compute_precision_recall(metrics):
    intersection = metrics[:, 0]
    union = metrics[:, 1]
    false_positive = metrics[:, 2]
    false_negative = metrics[:, 3]
    
    precision = intersection / (intersection + false_positive)
    recall = intersection / (intersection + false_negative)
    
    return precision, recall

precision_100, recall_100 = compute_precision_recall(metrics_100)
precision_200, recall_200 = compute_precision_recall(metrics_200)
precision_500, recall_500 = compute_precision_recall(metrics_500)

def compute_mean_std(data):
    return np.mean(data), np.std(data)

metrics_dict = {
    '100': (precision_100, recall_100),
    '200': (precision_200, recall_200),
    '500': (precision_500, recall_500)
}

densities = ['100', '200', '500']
density_values = [100, 200, 500]
precision_means = []
precision_stds = []
recall_means = []
recall_stds = []

for density in densities:
    precision, recall = metrics_dict[density]
    p_mean, p_std = compute_mean_std(precision)
    r_mean, r_std = compute_mean_std(recall)
    
    precision_means.append(p_mean)
    precision_stds.append(p_std)
    recall_means.append(r_mean)
    recall_stds.append(r_std)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,4),sharey=True)

ax1.errorbar(density_values, precision_means, yerr=precision_stds, fmt='x', 
             capsize=5, capthick=1, label='Precision', color='black')
ax1.set_xlabel(r'$\rho$',fontsize=16)
ax1.set_ylabel('Precision',fontsize=16)
ax1.grid()
ax1.set_xticks(density_values)

ax2.errorbar(density_values, recall_means, yerr=recall_stds, fmt='x', 
             capsize=5, capthick=1, label='Recall', color='black')
ax2.set_xlabel(r'$\rho$',fontsize=16)
ax2.set_ylabel('Recall',fontsize=16)
ax2.grid()
ax2.set_xticks(density_values)

plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, 'figure-2de.png'), dpi=300)
plt.show()

