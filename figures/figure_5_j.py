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
# Download files
# -----------------------------
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        r = requests.get(url)
        with open(filename, "wb") as f:
            f.write(r.content)

download_file("https://figshare.com/ndownloader/files/63422331", "ASD_gene.xlsx")
download_file("https://figshare.com/ndownloader/files/63422220", "KEGG_2021_Human.txt")
download_file("https://figshare.com/ndownloader/files/63422334", "ASD_predicted.json")

# -----------------------------
# Load Excel gene lists
# -----------------------------
sheets = pd.read_excel("ASD_gene.xlsx", sheet_name=None, header=None)

gene_lists = []
for name, df in sheets.items():
    print(f"Sheet: {name}")
    gene_lists.append(df[0].dropna().tolist())

# -----------------------------
# Load KEGG
# -----------------------------
gene_sets = {}
with open("KEGG_2021_Human.txt") as f:
    for line in f:
        parts = line.strip().split('\t')
        pathway = parts[0]
        genes = [g for g in parts[2:] if g]
        gene_sets[pathway] = genes

# -----------------------------
# First enrichment (ranking)
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

    for pathway in gene_sets:
        item = res[res['Term'] == pathway]
        if not item.empty:
            p = item['Adjusted P-value'].values[0]
            pathway_scores[pathway].append(-math.log10(p))
        else:
            pathway_scores[pathway].append(0)

# Select top pathways
avg_scores = {
    k: (sum(v)/len(v) if v else 0)
    for k, v in pathway_scores.items()
}

top_pathways = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:15]
important_GO = [x[0] for x in top_pathways]

# -----------------------------
# Load samples
# -----------------------------
with open("ASD_predicted.json") as f:
    samples = json.load(f)

# -----------------------------
# Second enrichment
# -----------------------------
result = []
number = []
numberb = []
buer = []

for sample in samples:

    groups = sample['nab']
    use_gene = groups[0] + groups[1] + groups[3]
    gene_mut = sample['mutation']

    number.append(len(use_gene))
    numberb.append(len(groups[0] + groups[1] + groups[2] + groups[3]))

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
def truncate_label(label, max_words=3):
    words = label.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + ' ...'
    return label

data = np.array(result)

base_height = 6
term_height = 0.3
fig_height = base_height + term_height * data.shape[0]
fontsize = 20

plt.figure(figsize=(12, fig_height))

ax = sns.heatmap(
    data,
    annot=True,
    fmt=".1f",
    cmap='YlOrRd',
    cbar=False,
    annot_kws={"size": fontsize},
    yticklabels=[f"{number[i]} of {numberb[i]}" for i in range(len(result))],
    xticklabels=[truncate_label(x) for x in important_GO],
    linewidths=0.5
)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=fontsize)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)

ax.set_xlabel("KEGG Terms", fontsize=fontsize)
ax.set_ylabel("Mutation-affected vs. total partner proteins", fontsize=fontsize)

# Highlight mutation-related cells
for text in ax.texts:
    row, col = map(int, text.get_position())
    if buer[col][row] == 1:
        text.set_color('#3b76fb')
        text.set_weight('bold')
        text.set_size(fontsize)

plt.tight_layout()
plt.savefig("ASD_hotplot_final.png", dpi=500)
plt.show()
