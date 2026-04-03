import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

url_csv = 'https://figshare.com/ndownloader/files/63423249'
df = pd.read_csv(url_csv)

label_def = df['label'].values
data_np = df.drop(columns=['label']).values

tsne = TSNE(n_components=2, random_state=42)
data_tsne = tsne.fit_transform(data_np)

plt.figure(figsize=(8.5, 8))
colors = ListedColormap(['#7E4909', '#E5CC8F', '#CCE5E5', '#0E8585'])
scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=label_def, cmap=colors, alpha=0.5)

labels = ['Disrupting', 'Decreasing', 'No effect', 'Increasing']
handles = [plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=colors(i / (len(labels)-1)), markersize=10) 
           for i in range(len(labels))]
plt.legend(handles, labels, title='Class Labels')

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(False)
plt.savefig('tsne_from_csv.png', dpi=300, bbox_inches='tight')
plt.show()
