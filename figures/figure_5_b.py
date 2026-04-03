import os
import requests
import pandas as pd
import gseapy as gp
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Download required files
# -----------------------------
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        r = requests.get(url)
        with open(filename, "wb") as f:
            f.write(r.content)

download_file(
    "https://figshare.com/ndownloader/files/63422223",
    "EE_gene.xlsx"
)
download_file(
    "https://figshare.com/ndownloader/files/63422220",
    "KEGG_2021_Human.txt"
)
download_file(
    "https://figshare.com/ndownloader/files/63422217",
    "EE_predicted.json"
)

# -----------------------------
# Load Excel data
# -----------------------------
file_path = 'EE_gene.xlsx'
sheets = pd.read_excel(file_path, sheet_name=None, header=None)
dataframes = {sheet_name: df for sheet_name, df in sheets.items()}

importants = []
for name, df in dataframes.items():
    print(f"Sheet: {name}")
    importants.append(df[0].tolist())

# -----------------------------
# Load KEGG gene sets
# -----------------------------
gene_sets = {}
with open("KEGG_2021_Human.txt", 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        pathway = parts[0]
        genes = [g for g in parts[2:] if g]
        gene_sets[pathway] = genes

# -----------------------------
# First enrichment pass
# -----------------------------
important_GO = [
    "Amino acid metabolism",
    "Neuroactive ligand-receptor interaction",
    "Cytokine-cytokine receptor interaction",
    "Calcium signaling pathway",
    "Synaptic vesicle cycle"
]

ssss = {key: [] for key in gene_sets.keys()}
result = []
number = []

for use_gene in importants:
    sample_result = []
    number.append(len(use_gene))

    enr = gp.enrichr(
        gene_list=use_gene,
        gene_sets=gene_sets,
        organism='human',
        outdir=None,
    )
    pandas_use = enr.results

    for GO in important_GO:
        item = pandas_use[pandas_use['Term'] == GO]
        if not item.empty:
            p_value = item['Adjusted P-value'].values[0]
            sample_result.append(-math.log10(p_value))
        else:
            sample_result.append(0)

    for GO in gene_sets.keys():
        item = pandas_use[pandas_use['Term'] == GO]
        if not item.empty:
            p_value = item['Adjusted P-value'].values[0]
            ssss[GO].append(-math.log10(p_value))
        else:
            ssss[GO].append(0)

    result.append(sample_result)

# -----------------------------
# Select top pathways
# -----------------------------
average_scores = {
    go: (sum(scores) / len(scores) if scores else 0)
    for go, scores in ssss.items()
}

top_go = sorted(
    average_scores.items(),
    key=lambda item: item[1],
    reverse=True
)[:14]

important_GO = []
for go_term, avg_score in top_go:
    print("GO term:", go_term, "Average score:", avg_score)
    important_GO.append(go_term)

# -----------------------------
# Load prediction data
# -----------------------------
with open('EE_predicted.json', 'r') as f:
    samples = json.load(f)

# -----------------------------
# Second enrichment pass
# -----------------------------
ssss = {key: [] for key in gene_sets.keys()}
result = []
number = []
numberb = []
buer = []

nnnn = 30

for idx, sample in enumerate(samples):
    if len(buer) == nnnn + 1:
        break

    sample_result = []
    samplep = sample['nab']
    use_gene = samplep[0] + samplep[1] + samplep[3]
    geneuse = sample['mutation']

    number.append(len(use_gene))
    numberb.append(len(samplep[0] + samplep[1] + samplep[2] + samplep[3]))

    if len(use_gene) == 0:
        result.append([0 for _ in range(len(important_GO))])

        sample_buer = []
        for GO in important_GO:
            sample_buer.append(1 if geneuse in gene_sets[GO] else 0)
        buer.append(sample_buer)
        continue

    enr = gp.enrichr(
        gene_list=use_gene,
        gene_sets=gene_sets,
        organism='human',
        outdir=None,
    )

    pandas_use = enr.results

    if len(pandas_use) == 0:
        result.append([0 for _ in range(len(important_GO))])

        sample_buer = []
        for GO in important_GO:
            sample_buer.append(1 if geneuse in gene_sets[GO] else 0)
        buer.append(sample_buer)
        continue

    for GO in important_GO:
        item = pandas_use[pandas_use['Term'] == GO]
        if not item.empty:
            p_value = item['Adjusted P-value'].values[0]
            sample_result.append(-math.log10(p_value))
        else:
            sample_result.append(0)

    for GO in gene_sets.keys():
        item = pandas_use[pandas_use['Term'] == GO]
        if not item.empty:
            p_value = item['Adjusted P-value'].values[0]
            ssss[GO].append(-math.log10(p_value))
        else:
            ssss[GO].append(0)

    result.append(sample_result)

    sample_buer = []
    for GO in important_GO:
        sample_buer.append(1 if geneuse in gene_sets[GO] else 0)
    buer.append(sample_buer)

    if len(buer) == nnnn:
        break

# -----------------------------
# Plot heatmap
# -----------------------------
start = 0
data = np.array(result[start:])

base_height = 6
term_height = 0.3
dynamic_height = base_height + term_height * data.shape[0]
fontsize = 15

plt.figure(figsize=(12, dynamic_height))

ax = sns.heatmap(
    data,
    annot=True,
    fmt=".1f",
    cmap='YlOrRd',
    cbar=False,
    yticklabels=[f"{number[i]} of {numberb[i]}" for i in range(len(result[start:]))],
    xticklabels=important_GO[start:],
    linewidths=0.5
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    ha='right',
    fontsize=fontsize
)

ax.set_yticklabels(
    ax.get_yticklabels(),
    fontsize=fontsize,
    rotation=0
)

plt.xlabel("KEGG Terms", labelpad=fontsize, fontsize=fontsize)
plt.ylabel("Mutation-affected vs. total partner proteins", labelpad=fontsize, fontsize=fontsize)

for text in ax.texts:
    row, col = map(int, text.get_position())
    if buer[start:][col][row] == 1:
        text.set_color('#3b76fb')
        text.set_weight('bold')

plt.rcParams['svg.fonttype'] = 'none'

plt.tight_layout()
plt.savefig("hotplot_TAPPI_EE.svg", format="svg", bbox_inches="tight")
plt.show()
plt.close()