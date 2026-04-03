import os
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt

# Download file
url = "https://figshare.com/ndownloader/files/63410256"
file_path = "Source_Data_EE.xlsx"

if not os.path.exists(file_path):
    r = requests.get(url)
    with open(file_path, "wb") as f:
        f.write(r.content)

# Load Excel
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names

# Infer grid size
a = len(sheet_names) // 4
b = 4

# Plot settings
sns.set(style="whitegrid", font_scale=2)

fig = plt.figure(figsize=(5*b, 5*a + 2))
gs = fig.add_gridspec(nrows=a+1, ncols=b, height_ratios=[*[5]*a, 2])

axes = []
for i in range(a):
    row_axes = []
    for j in range(b):
        ax = fig.add_subplot(gs[i, j])
        row_axes.append(ax)
    axes.append(row_axes)

legend_ax = fig.add_subplot(gs[a, :])
legend_ax.axis('off')

dx = "distance"
dy = "proportion"
dhue = "phenotype"

palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
labels_mippi = ["Disrupting", "Decreasing", "No_effect", "Increasing"]

handles, labels = [], []

for i in range(a):
    for j in range(4):
        sheet_name = f"class_{i+1}_type_{j+1}"
        df = pd.read_excel(xls, sheet_name=sheet_name)

        ax = axes[i][j]

        pt.RainCloud(
            x=dx, y=dy, hue=dhue, data=df,
            palette=palette,
            bw=.2,
            width_viol=.7,
            orient="h",
            alpha=.65,
            dodge=True,
            pointplot=True,
            move=.2,
            ax=ax
        )

        ax.set_xlim(0, 1)
        ax.set_title(f"class {i+1}, {labels_mippi[j]}")
        ax.legend().remove()

        if i == 0 and j == 0:
            handles, labels = ax.get_legend_handles_labels()

legend_ax.legend(
    handles[:2],
    labels[:2],
    title="Phenotype",
    ncol=2,
    loc='center',
    frameon=False
)

plt.tight_layout()
plt.savefig("Figure_EE_full.png", dpi=500, bbox_inches='tight')
plt.show()