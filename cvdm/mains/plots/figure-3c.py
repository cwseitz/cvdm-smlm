import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path='/N/slate/cwseitz/cvdm/Sim/4x/N-100_N0-500-1000/eval_data/'
df100=pd.read_csv(path+'N100-error.csv')
path='/N/slate/cwseitz/cvdm/Sim/4x/N-200_N0-500-1000/eval_data/'
df200=pd.read_csv(path+'N200-error.csv')
path='/N/slate/cwseitz/cvdm/Sim/4x/N-500_N0-500-1000/eval_data/'
df500=pd.read_csv(path+'N500-error.csv')

def compute_stats(df):
    grouped=df.groupby(['label','prefix','idx'])
    xstd=[np.std(group['x_err'].values) for _,group in grouped]
    ystd=[np.std(group['y_err'].values) for _,group in grouped]
    return np.array(xstd),np.array(ystd)

xstd_100, ystd_100 = compute_stats(df100)
xstd_200, ystd_200 = compute_stats(df200)
xstd_500, ystd_500 = compute_stats(df500)

xstd_100 = xstd_100[xstd_100 > 0]
ystd_100 = ystd_100[ystd_100 > 0]
xstd_200 = xstd_200[xstd_200 > 0]
ystd_200 = ystd_200[ystd_200 > 0]
xstd_500 = xstd_500[xstd_500 > 0]
ystd_500 = ystd_500[ystd_500 > 0]

pixel_size = 25.0
datasets = [
    (xstd_100 * pixel_size, ystd_100 * pixel_size),
    (xstd_200 * pixel_size, ystd_200 * pixel_size),
    (xstd_500 * pixel_size, ystd_500 * pixel_size)
]

labels = [r'$\rho=100$', r'$\rho=200$', r'$\rho=500$']

all_values = np.concatenate([ds for pair in datasets for ds in pair])
x_min, x_max = np.min(all_values), np.max(all_values)

fig, axes = plt.subplots(2, 3, figsize=(8, 4), sharey='row', sharex='col')
colors = ['red', 'blue', 'gray']

for i, (xstd, ystd) in enumerate(datasets):
    for ax, std, label in zip(axes[:, i], [xstd, ystd], 
                              [r'$\sqrt{\mathrm{Var}(\theta_u)}$', r'$\sqrt{\mathrm{Var}(\theta_v)}$']):
        hist, bins = np.histogram(std, bins=20, density=True)
        ax.bar(bins[:-1], hist, width=np.diff(bins), color=colors[i], edgecolor='black')
        mean_value = np.mean(std)
        ax.axvline(mean_value, color='black', linestyle='--', label=f'Mean = {mean_value:.2f}')
        if ax.get_subplotspec().is_first_row():
            ax.set_title(labels[i], fontsize=14)
        ax.set_xlabel(label + ' (nm)', fontsize=12)
        ax.set_xlim(x_min, x_max)
        ax.legend(frameon=False, fontsize=10)

axes[0, 0].set_ylabel('Probability', fontsize=12)
axes[1, 0].set_ylabel('Probability', fontsize=12)

plt.tight_layout()
exp = '/projects/data_solutions/raiders/pathai/seitzcx/exps/exp_CVDM/results'
plt.savefig(exp+'figure-3c.png',dpi=300)
plt.show()

