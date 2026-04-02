import os
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt

url = "https://figshare.com/ndownloader/files/63410256"
file_path = "Source_Data_EE.xlsx"

if not os.path.exists(file_path):
    r = requests.get(url)
    with open(file_path, "wb") as f:
        f.write(r.content)

xls = pd.ExcelFile(file_path)

df = pd.read_excel(xls, sheet_name="class_1_type_3")

sns.set(style="whitegrid", font_scale=2)

plt.figure(figsize=(8, 6))

pt.RainCloud(
    x="distance",
    y="proportion",
    hue="phenotype",
    data=df,
    palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    bw=.2,
    width_viol=.7,
    orient="h",
    alpha=.65,
    dodge=True,
    pointplot=True,
    move=.2
)

plt.xlim(0, 1)
plt.title("Class 1 - No_effect")

# plt.legend(title="Phenotype", frameon=False)

plt.tight_layout()
plt.savefig("Figure_EE_class1_type3.png", dpi=500, bbox_inches='tight')
plt.show()