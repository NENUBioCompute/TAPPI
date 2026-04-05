import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1. Download data
# -----------------------------
url = "https://figshare.com/ndownloader/files/63428655" 
local_file = "ddg_data.csv"

r = requests.get(url)
r.raise_for_status()

with open(local_file, "wb") as f:
    f.write(r.content)

print(f"Downloaded {local_file}")

# -----------------------------
# 2. Load data
# -----------------------------
# Expecting columns: label_ddg, pred_ddg
data = pd.read_csv(local_file)

print(data.head())

# -----------------------------
# 3. Plot violin
# -----------------------------
sns.set(style="whitegrid", font_scale=1.5)

plt.figure(figsize=(10, 8))

sns.violinplot(
    x='pred_ddg',
    y='label_ddg',
    data=data,
    palette=['#7E4909', '#E5CC8F', '#CCE5E5', '#0E8585']
)

plt.xlabel('Effect Type', fontsize=18)
plt.ylabel('ΔΔG Values', fontsize=18)

plt.xticks(
    ticks=[0, 1, 2, 3],
    labels=['disrupting', 'decreasing', 'no effect', 'increasing'],
    fontsize=18
)

plt.yticks(fontsize=18)
plt.grid()

plt.savefig('plot_DDG.png', dpi=300, bbox_inches='tight')
plt.show()

import matplotlib.pyplot as plt
import pandas as pd

labels = ['Disrupting', 'Decreasing', 'No effect', 'Increasing']

df = data

subset_pos = df[df['label_ddg'] > 1]['pred_ddg']
subset_neg = df[df['label_ddg'] < -1]['pred_ddg']


values_pos = (
    subset_pos.value_counts(normalize=True)
    .reindex([0,1,2,3], fill_value=0)
    .values
)

values_neg = (
    subset_neg.value_counts(normalize=True)
    .reindex([0,1,2,3], fill_value=0)
    .values
)


color_pos = '#a7c0df'
color_neg = '#a0d0d0'


fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

ax = axes[0]
bars = ax.bar(labels, values_pos, color=color_pos, width=0.6)
ax.set_title('ddG > 1', fontsize=14)
ax.set_ylabel('Proportion', fontsize=12)
ax.set_ylim(0, 1.0)
for bar, v in zip(bars, values_pos):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
            f'{v:.2f}', ha='center', va='bottom', fontsize=11)
ax.tick_params(axis='x', rotation=45)

ax = axes[1]
bars = ax.bar(labels, values_neg, color=color_neg, width=0.6)
ax.set_title('ddG < -1', fontsize=14)
ax.set_ylim(0, 1.0)
for bar, v in zip(bars, values_neg):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
            f'{v:.2f}', ha='center', va='bottom', fontsize=11)
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.rcParams['svg.fonttype'] = 'none'

plt.savefig("DDG.png", format="png", bbox_inches="tight", dpi=500)
plt.show()
plt.close()
