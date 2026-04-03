import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt

# -----------------------------
# 1. Download the Excel file
# -----------------------------
url = "https://figshare.com/ndownloader/files/63430029"
local_file = "raincloud_edge_halfquater.xlsx"

r = requests.get(url)
r.raise_for_status()

with open(local_file, "wb") as f:
    f.write(r.content)

print(f"Downloaded {local_file} from Figshare")

# -----------------------------
# 2. Load Excel sheets
# -----------------------------
xls = pd.ExcelFile(local_file)

sheets = xls.sheet_names
print("Found sheets:", sheets)

# Group sheets into classes (each class has 4 types)
dataframes = []
num_types = 4
num_classes = len(sheets) // num_types

for i in range(num_classes):
    group = []
    for j in range(num_types):
        sheet_name = f"class_{i+1}_type_{j+1}"
        df = pd.read_excel(xls, sheet_name=sheet_name)
        group.append(df)
    dataframes.append(group)

print(f"Loaded {len(dataframes)} classes with {num_types} types each")

# -----------------------------
# 3. Plot RainCloud figure
# -----------------------------
sns.set(style="whitegrid", font_scale=2)

a = len(dataframes)
b = len(dataframes[0])

fig = plt.figure(figsize=(5*b, 5*a + 2))
gs = fig.add_gridspec(nrows=a+1, ncols=b, height_ratios=[*[5]*a, 2])

axes = []
for i in range(a):
    row_axes = []
    for j in range(b):
        ax = fig.add_subplot(gs[i, j])
        row_axes.append(ax)
    axes.append(row_axes)

# Legend axis
legend_ax = fig.add_subplot(gs[a, :])
legend_ax.axis('off')

dx = "distance"
dy = "proportion"
dhue = "phenotype"
ort = "h"

pal = [
    "#D82F25", "#4E1945", "#CB9475", "#8CBF87",
    "#3E608D", "#909291", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf"
]

sigma = 0.2
labels_mippi = ["Disrupting", "Decreasing", "No_effect", "Increasing"]

handles, labels = [], []

for i, df_group in enumerate(dataframes):
    for j, part_df in enumerate(df_group):
        ax = axes[i][j]

        ax = pt.RainCloud(
            x=dx, y=dy, hue=dhue, data=part_df,
            palette=pal, bw=sigma,
            width_viol=.7, ax=ax, orient=ort,
            alpha=.65, dodge=True, pointplot=True, move=.2
        )

        ax.set_xlim(0, 1)
        ax.set_title(f"class {i+1}, {labels_mippi[j]}")
        ax.legend().remove()

        if i == 0 and j == 0:
            handles, labels = ax.get_legend_handles_labels()

# Remove duplicate labels
labels_unique = list(dict.fromkeys(labels))
handles_show = handles[:len(labels_unique)]

legend_ax.legend(
    handles_show, labels_unique,
    title="Phenotype",
    ncol=len(labels_unique),
    loc='center',
    frameon=False,
    fontsize=16,
    title_fontsize=18
)

plt.subplots_adjust(hspace=0.5, wspace=0.5)

plt.show()
