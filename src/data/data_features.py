# ============================================================
# Adding New Features
# ============================================================

# 1. PCA (Principal Component Analysis)
# 2. Kmeans

# ============================================================
# Libraries Import
# ============================================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

plt.style.use("seaborn-v0_8-colorblind")
plt.rcParams["figure.figsize"] = (15, 12)
plt.rcParams["figure.dpi"] = 100

sns.set_style("whitegrid")

# ============================================================
# Load Data
# ============================================================


df = pd.read_csv("../../data/processed/df_with_no_outliers.csv")

# ============================================================
# Applying PCA
# ============================================================

features_to_pca0 = [feature for feature in df.columns if feature.startswith("num")]
features_to_pca1 = [
    feature
    for feature in df.columns
    if feature.startswith("valor") and feature != "valor_aluguel"
]

scaler = StandardScaler()

pca = PCA(n_components=2, random_state=0)
pca.set_output(transform="pandas")

num_pca_features = pca.fit_transform(scaler.fit_transform(df[features_to_pca0]))
valor_pca_features = pca.fit_transform(scaler.fit_transform(df[features_to_pca1]))

df[["num_pca_0", "num_pca_1"]] = num_pca_features
df[["valor_pca_0", "valor_pca_1"]] = valor_pca_features


# ============================================================
# Clustering
# ============================================================
df_cluster = df.copy()
cluster_columns = ["valor_condominio", "valor_iptu", "valor_seguro_incendio"]
n_clusters = range(2, 20)
inertias = []

for n in n_clusters:
    cluster = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=n, random_state=0)
    cluster_labels = kmeans.fit_predict(scaler.fit_transform(cluster))
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(4, 4))
plt.plot(n_clusters, inertias)
plt.xlabel("Clusters Number")
plt.ylabel("Sum of Squared Distances")
plt.savefig("../../reports/clustering_inertias.jpg")
plt.show()

kmeans = KMeans(n_clusters=6, random_state=0)
cluster_labels = kmeans.fit_predict(scaler.fit_transform(df_cluster[cluster_columns]))
df_cluster["cluster"] = cluster_labels

plt.scatter(data=df_cluster, x="area", y="valor_aluguel", c="cluster")
plt.xlabel("Área")
plt.ylabel("Valor do aluguel")
plt.savefig(
    "../../reports/clustering_scatter.jpg",
)
plt.show()

plt.figure(figsize=(9, 7))
sns.countplot(
    x=df_cluster["cluster"],
    hue=df_cluster["cluster"],
    palette="viridis",
    width=0.5,
    legend=True,
)

plt.xlabel("Clusters")
plt.savefig(
    "../../reports/clustering_distribution.jpg",
)
plt.show()

# ============================================================
# Export Dataset
# ============================================================
df_cluster.to_csv("../../data/processed/df_with_new_features.csv", index=None)
