import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import ScalarFormatter, MultipleLocator

BASE_PATH = '/Users/seitzcx/Desktop/exps/exp_CVDM/results/errors_sgra'

prefixes_100 = [
'cvdm_sz64_n1000_N100-101_0bb3fc88-error.csv',
# 'cvdm_sz64_n1000_N100-101_2cefe591-error.csv',
# 'cvdm_sz64_n1000_N100-101_6775a814-error.csv',
# 'cvdm_sz64_n1000_N100-101_6fea8ebc-error.csv',
# 'cvdm_sz64_n1000_N100-101_8ba68f8c-error.csv',
# 'cvdm_sz64_n1000_N100-101_a2900756-error.csv',
# 'cvdm_sz64_n1000_N100-101_c3b595da-error.csv',
# 'cvdm_sz64_n1000_N100-101_d983656b-error.csv',
# 'cvdm_sz64_n1000_N100-101_e1a90762-error.csv'
]

prefixes_200 = [
'cvdm_sz64_n1000_N200-201_08e0cb8f-error.csv',
# 'cvdm_sz64_n1000_N200-201_1064a130-error.csv',
# 'cvdm_sz64_n1000_N200-201_17711672-error.csv',
# 'cvdm_sz64_n1000_N200-201_4b994678-error.csv',
# 'cvdm_sz64_n1000_N200-201_5e863f28-error.csv',
# 'cvdm_sz64_n1000_N200-201_6345e77e-error.csv',
# 'cvdm_sz64_n1000_N200-201_73d97b21-error.csv',
# 'cvdm_sz64_n1000_N200-201_9b63ec82-error.csv',
# 'cvdm_sz64_n1000_N200-201_ddafbc59-error.csv',
# 'cvdm_sz64_n1000_N200-201_f27efd73-error.csv'
]
prefixes_500 = [
'cvdm_sz64_n1000_N500-501_08579b5a-error.csv',
# 'cvdm_sz64_n1000_N500-501_1ca31d33-error.csv',
# 'cvdm_sz64_n1000_N500-501_3277fdae-error.csv',
# 'cvdm_sz64_n1000_N500-501_422ca18a-error.csv',
# 'cvdm_sz64_n1000_N500-501_4e13f5e4-error.csv',
# 'cvdm_sz64_n1000_N500-501_6b94f127-error.csv',
# 'cvdm_sz64_n1000_N500-501_7120a4e5-error.csv',
# 'cvdm_sz64_n1000_N500-501_867a95c3-error.csv',
# 'cvdm_sz64_n1000_N500-501_c9acfc1c-error.csv',
# 'cvdm_sz64_n1000_N500-501_e1305d0b-error.csv'
]

def load_prefixes(prefixes):
    if not prefixes:
        return pd.DataFrame()
    dfs = [pd.read_csv(os.path.join(BASE_PATH, prefix)) for prefix in prefixes]
    return pd.concat(dfs, ignore_index=True)

print('Loading prefixes...')
df100 = load_prefixes(prefixes_100)
df200 = load_prefixes(prefixes_200)
df500 = load_prefixes(prefixes_500)
print(df100)
print(df200)
print(df500)

def compute_stats(df, label=""):
    if df.empty:
        print(f"Skipping stats {label}: empty dataframe")
        return np.array([]), np.array([]), np.array([])
    print(f"Computing stats {label}... (rows={len(df)})")
    grouped = df.groupby(['label', 'prefix', 'idx'], sort=False, observed=True)
    stats = grouped.agg(x_err_mean=('x_err', 'mean'), y_err_mean=('y_err', 'mean'), N0=('N0', 'first'))
    print(f"Computed stats {label}: groups={len(stats)}")
    return stats['N0'].to_numpy(), stats['x_err_mean'].to_numpy(), stats['y_err_mean'].to_numpy()

N0_100, xacc_100, yacc_100 = compute_stats(df100, "rho=100")
N0_200, xacc_200, yacc_200 = compute_stats(df200, "rho=200")
N0_500, xacc_500, yacc_500 = compute_stats(df500, "rho=500")

pixel_size = 25.0 #nm
bins = np.arange(500, 1000, 100)

def bin_std(N0, acc):
    if N0.size == 0 or acc.size == 0:
        return np.array([])
    return np.array([np.std(acc[(N0 >= bins[i]) & (N0 < bins[i+1])]) for i in range(len(bins) - 1)])

print('Binning stdevs...')
xstd_binned_100 = bin_std(N0_100, xacc_100)
ystd_binned_100 = bin_std(N0_100, yacc_100)
xstd_binned_200 = bin_std(N0_200, xacc_200)
ystd_binned_200 = bin_std(N0_200, yacc_200)
xstd_binned_500 = bin_std(N0_500, xacc_500)
ystd_binned_500 = bin_std(N0_500, yacc_500)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

if xstd_binned_100.size:
    ax1.plot(bins[:-1], xstd_binned_100 * pixel_size, 'x', color='red', label=r'$\rho = 100$', markersize=7)
if xstd_binned_200.size:
    ax1.plot(bins[:-1], xstd_binned_200 * pixel_size, 'x', color='blue', label=r'$\rho = 200$', markersize=7)
if xstd_binned_500.size:
    ax1.plot(bins[:-1], xstd_binned_500 * pixel_size, 'x', color='gray', label=r'$\rho = 500$', markersize=7)

ax1.set_xscale('log')
ax1.set_xticks(bins[:-1])
ax1.set_xlabel('Photons', fontsize=16)
ax1.set_ylabel(r'$\sigma_{u}$ (nm)', fontsize=16)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid()

ax1.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.xaxis.get_major_formatter().set_scientific(True)
ax1.xaxis.get_major_formatter().set_powerlimits((0, 0))
ax1.yaxis.set_major_locator(MultipleLocator(5))

if ystd_binned_100.size:
    ax2.plot(bins[:-1], ystd_binned_100 * pixel_size, 'x', color='red', label=r'$\rho = 100$', markersize=7)
if ystd_binned_200.size:
    ax2.plot(bins[:-1], ystd_binned_200 * pixel_size, 'x', color='blue', label=r'$\rho = 200$', markersize=7)
if ystd_binned_500.size:
    ax2.plot(bins[:-1], ystd_binned_500 * pixel_size, 'x', color='gray', label=r'$\rho = 500$', markersize=7)

ax2.set_xscale('log')
ax2.set_xticks(bins[:-1])
ax2.set_xlabel('Photons', fontsize=16)
ax2.set_ylabel(r'$\sigma_{v}$ (nm)', fontsize=16)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax2.xaxis.get_major_formatter().set_scientific(True)
ax2.xaxis.get_major_formatter().set_powerlimits((0, 0))
ax2.yaxis.set_major_locator(MultipleLocator(5))

ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False,fontsize=12)
ax2.grid()
plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, 'figure-2bc.png'), dpi=300)
plt.show()

# Histogram of errors for 100/200/500
def collect_errors(df, label=""):
    if df.empty:
        print(f"Skipping histogram {label}: empty dataframe")
        return np.array([])
    if 'x_err' not in df.columns or 'y_err' not in df.columns:
        print(f"Skipping histogram {label}: missing x_err/y_err")
        return np.array([])
    return np.concatenate([df['x_err'].to_numpy(), df['y_err'].to_numpy()]) * pixel_size

err_100 = collect_errors(df100, "rho=100")
err_200 = collect_errors(df200, "rho=200")
err_500 = collect_errors(df500, "rho=500")

hist_bins = 60
fig_hist, ax_hist = plt.subplots(figsize=(2,2), dpi=300)

if err_100.size:
    ax_hist.hist(err_100, bins=hist_bins, alpha=0.6, color='red', label=r'$\rho = 100$', density=True)
if err_200.size:
    ax_hist.hist(err_200, bins=hist_bins, alpha=0.6, color='blue', label=r'$\rho = 200$', density=True)
if err_500.size:
    ax_hist.hist(err_500, bins=hist_bins, alpha=0.6, color='gray', label=r'$\rho = 500$', density=True)

ax_hist.set_xlabel('Error (nm)', fontsize=10)
ax_hist.set_ylabel('Density', fontsize=10)
ax_hist.spines['top'].set_visible(False)
ax_hist.spines['right'].set_visible(False)
ax_hist.grid(alpha=0.3)
ax_hist.legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, 'figure-2bc-hist.png'), dpi=300)
plt.show()

