import requests
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math

# -----------------------------
# 1. Download data
# -----------------------------
url = "https://figshare.com/ndownloader/files/63426651"
local_file = "violin_data.json"

r = requests.get(url)
r.raise_for_status()

with open(local_file, "wb") as f:
    f.write(r.content)

print(f"Downloaded {local_file}")

# -----------------------------
# 2. Load data
# -----------------------------
with open(local_file, "r") as f:
    data = json.load(f)

list_A_1 = data["list_A_1"]
list_B_1 = data["list_B_1"]
list_C_1 = data["list_C_1"]
highlight_indices = data["highlight_indices"]

# -----------------------------
# 3. Build dataframe
# -----------------------------
df = pd.DataFrame({
    "value_log": list_A_1 + list_C_1 + list_B_1,
    "group": ["TAPPI"] * len(list_A_1) +
             ["PLM_interact"] * len(list_C_1) +
             ["mint"] * len(list_B_1)
})

# Highlight values
highlight_values = [list_A_1[i] for i in highlight_indices]

# Remove highlighted points from strip layer
df_strip = df.drop(index=highlight_indices)

palette = {
    "TAPPI": "#9BBBE1",
    "PLM_interact": "#B7B7EB",
    "mint": "#9D9EA3"
}

# -----------------------------
# 4. Plot
# -----------------------------
plt.figure(figsize=(6, 8))

ax = sns.violinplot(
    data=df,
    x="group",
    y="value_log",
    palette=palette,
    cut=0,
    linewidth=0.8,
    inner=None
)

sns.stripplot(
    data=df_strip,
    x="group",
    y="value_log",
    color="black",
    size=1.2,
    jitter=0.25,
    alpha=0.15,
    ax=ax
)

sns.pointplot(
    data=df,
    x="group",
    y="value_log",
    estimator="median",
    color="black",
    markers="_",
    scale=1.2,
    linestyles="",
    ax=ax
)

# Highlight points
jitter = np.random.uniform(-0.05, 0.05, len(highlight_values))
ax.scatter(
    0 + jitter,
    highlight_values,
    color="#D62728",
    s=80,
    edgecolor="white",
    linewidth=1.2,
    zorder=7
)

# Restore y-axis scale
yticks = ax.get_yticks()
ax.set_yticklabels([f"{math.exp(y)-1:.2f}" for y in yticks])

ax.set_ylabel("Value")
ax.set_xlabel("")

sns.despine()
plt.rcParams['svg.fonttype'] = 'none'

plt.tight_layout()
plt.savefig("violin_plot.svg", format="svg", bbox_inches="tight")
plt.show()
plt.close()
