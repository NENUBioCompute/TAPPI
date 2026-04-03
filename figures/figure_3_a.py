import matplotlib.pyplot as plt
import json
import requests

# Download the JSON file from Figshare
url = "https://figshare.com/ndownloader/files/63423843"
r = requests.get(url)
with open("attention_data.json", "wb") as f:
    f.write(r.content)

# Load saved data
with open("attention_data.json", "r") as f:
    data = json.load(f)

y = data["attention"]
highlighted_indices = data["highlighted"]

x = range(len(y))

plt.figure(figsize=(8, 3))
plt.plot(x, y, linestyle='-', color='#315A89', label='Attention Weight')

# Merge consecutive indices for highlighting
merged_ranges = []
for sublist in highlighted_indices:
    current_range = [sublist[0]]
    for i in sublist[1:]:
        if i == current_range[-1] + 1:
            current_range.append(i)
        else:
            merged_ranges.append(current_range)
            current_range = [i]
    merged_ranges.append(current_range)

# Draw highlighted regions
for range_ in merged_ranges:
    plt.axvspan(range_[0] - 0.5, range_[-1] + 0.5, color='#E5CC8F', alpha=0.5)

fontsize = 18
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
handles = [
    plt.Line2D([0], [0], color='#315A89', lw=2, label='Attention Weight'),
    plt.Line2D([0], [0], color='#E5CC8F', lw=10, label='Binding Region', alpha=0.5)
]
plt.legend(handles=handles, fontsize=fontsize)
plt.savefig('attention_plot_from_figshare.png', bbox_inches='tight', dpi=300)
plt.show()
