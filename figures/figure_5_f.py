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

download_file("https://figshare.com/ndownloader/files/63422283", "LGS_gene.xlsx")
download_file("https://figshare.com/ndownloader/files/63422220", "KEGG_2021_Human.txt")
download_file("https://figshare.com/ndownloader/files/63422286", "LGS_predicted.json")

# -----------------------------
# Load gene lists from Excel
# -----------------------------
sheets = pd.read_excel("LGS_gene.xlsx", sheet_name=None, header=None)

gene_lists = []
for name, df in sheets.items():
    print(f"Sheet: {name}")
    gene_lists.append(df[0].dropna().tolist())

# -----------------------------
# Load KEGG gene sets
# -----------------------------
gene_sets = {}
with open("KEGG_2021_Human.txt") as f:
    for line in f:
        parts = line.strip().split('\t')
        pathway = parts[0]
        genes = [g for g in parts[2:] if g]
        gene_sets[pathway] = genes

# -----------------------------
# First enrichment (for ranking)
# -----------------------------
pathway_scores = {k: [] for k in gene_sets.keys()}

for genes in gene_lists:
    enr = gp.enrichr(
        gene_list=genes,
        gene_sets=gene_sets,
        organism='human',
        outdir=None
    )
    res = enr.results

    for pathway in gene_sets.keys():
        item = res[res['Term'] == pathway]
        if not item.empty:
            p = item['Adjusted P-value'].values[0]
            pathway_scores[pathway].append(-math.log10(p))
        else:
            pathway_scores[pathway].append(0)

# Average score
avg_scores = {
    k: (sum(v)/len(v) if v else 0)
    for k, v in pathway_scores.items()
}

# Select top pathways
top_pathways = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:12]
important_GO = [x[0] for x in top_pathways]

print("\nTop pathways:")
for k, v in top_pathways:
    print(k, v)

# -----------------------------
# Load sample data
# -----------------------------
with open("LGS_predicted.json") as f:
    samples = json.load(f)

# -----------------------------
# Second enrichment (per sample)
# -----------------------------
result = []
number = []
numberb = []
buer = []

max_samples = 30

for idx, sample in enumerate(samples):
    if len(result) >= max_samples:
        break

    groups = sample['nab']
    use_gene = groups[0] + groups[1] + groups[3]
    gene_mut = sample['mutation']

    number.append(len(use_gene))
    numberb.append(len(groups[0] + groups[1] + groups[2] + groups[3]))

    # Handle empty gene list
    if len(use_gene) == 0:
        result.append([0]*len(important_GO))
        buer.append([1 if gene_mut in gene_sets[g] else 0 for g in important_GO])
        continue

    enr = gp.enrichr(
        gene_list=use_gene,
        gene_sets=gene_sets,
        organism='human',
        outdir=None
    )

    res = enr.results

    if len(res) == 0:
        result.append([0]*len(important_GO))
        buer.append([1 if gene_mut in gene_sets[g] else 0 for g in important_GO])
        continue

    row = []
    for g in important_GO:
        item = res[res['Term'] == g]
        if not item.empty:
            p = item['Adjusted P-value'].values[0]
            row.append(-math.log10(p))
        else:
            row.append(0)

    result.append(row)
    buer.append([1 if gene_mut in gene_sets[g] else 0 for g in important_GO])

# -----------------------------
# Plot heatmap
# -----------------------------
data = np.array(result)

base_height = 6
term_height = 0.3
fig_height = base_height + term_height * data.shape[0]

plt.figure(figsize=(12, fig_height))

ax = sns.heatmap(
    data,
    annot=True,
    fmt=".1f",
    cmap='YlOrRd',
    cbar=False,
    yticklabels=[f"{number[i]} of {numberb[i]}" for i in range(len(result))],
    xticklabels=important_GO,
    linewidths=0.5
)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=15)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)

plt.xlabel("KEGG Terms", fontsize=15)
plt.ylabel("Mutation-affected vs. total partner proteins", fontsize=15)

# Highlight mutation-related cells
for text in ax.texts:
    row, col = map(int, text.get_position())
    if buer[col][row] == 1:
        text.set_color('#3b76fb')
        text.set_weight('bold')

plt.tight_layout()
plt.savefig("LGS_hotplot.png", dpi=500)
plt.show()