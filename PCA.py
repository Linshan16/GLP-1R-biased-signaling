import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm

data_path = "data/mydata"
data_filename = "esm2_GLP1R_deepmut_embed.csv"
out_path = "deepmut"
n_components = 10


data_df = pd.read_csv(f"{data_path}/{data_filename}")
data_array = np.array(data_df.iloc[0:, 1:])
group_labels = data_df["mut"].values.tolist()
for i in range(len(group_labels)):
    group_labels[i] = group_labels[i][1:-1]
    if group_labels[i] not in ["121", "89", "68", "91"]:
        if group_labels[i] == "":
            group_labels[i] = "WT"
        else:
            group_labels[i] = "other"

# Define colors for each group label
color_mapping = {'121': 'blue', '89': 'red', '68': 'yellow', '91': 'green', "WT": "black", "other": "lightgray"}
numeric_labels = np.array([color_mapping[label] for label in group_labels])
cmap = cm.get_cmap('viridis')


# Initialize PCA with the desired number of components
pca = PCA(n_components=n_components)

# Fit PCA on the embedding data
embedding_pca = pca.fit_transform(data_array)
pca_embed_df = pd.DataFrame(embedding_pca, columns=range(n_components))
pca_embed_df.insert(0, "mut", data_df["mut"])
pca_embed_df.to_csv(f"{data_path}/{out_path}/pca_{n_components}_mut_embed.csv", index=None)

# Display the composition of the selected principal component
pca_compos_df = pd.DataFrame(pca.components_, columns=[f"Feature_{i}" for i in range(1, data_array.shape[1] + 1)], index=[f"PC{i}" for i in range(1, n_components + 1)])
pca_compos_df.to_csv(f"{data_path}/{out_path}/pca_{n_components}_component_composition.csv")

# Print the explained variance ratio for each component
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance_ratio)

# Plot 1: Percentage Deviation of Each Component
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.bar(range(1, n_components + 1), explained_variance_ratio * 100)
plt.title('Percentage Deviation of Each Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Percentage Deviation')

# Plot 2: Scatter plot of the first two components, colored by group label
plt.subplot(1, 2, 2)
scatter = plt.scatter(embedding_pca[:, 0], embedding_pca[:, 1], c=numeric_labels, cmap=cmap, norm=norm, alpha=0.8, edgecolors=[color_mapping[label] for label in group_labels], linewidths=0.5)
cbar = plt.colorbar(scatter, ticks=np.linspace(min(numeric_labels), max(numeric_labels), 10))
cbar.set_label('Numeric Group Label')
plt.title('Scatter Plot of the First Two Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.savefig(f"{data_path}/{out_path}/pca_{n_components}.png")
plt.show()
