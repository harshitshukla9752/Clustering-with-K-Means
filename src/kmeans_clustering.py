# src/kmeans_clustering.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ----------------------------
# 1. Load Dataset
# ----------------------------
data = pd.read_csv(r"C:\Users\harsh\python-projects\Clustering-with-K-Means\data\Mall_Customers.csv")  # Use your dataset path
print("Dataset shape:", data.shape)
print("Columns:", data.columns)

# ----------------------------
# 2. Preprocessing
# ----------------------------
# Encode 'Gender' as numeric
data['Gender'] = data['Gender'].map({'Female':0, 'Male':1})

# Features for clustering
X = data[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 3. Optional: Visualize raw data (scatter pairplot)
# ----------------------------
sns.pairplot(data[['Age','Annual Income (k$)','Spending Score (1-100)','Gender']], hue='Gender')
plt.savefig('outputs/raw_data_pairplot.png')
plt.close()
print("Raw data pairplot saved to outputs/raw_data_pairplot.png")

# ----------------------------
# 4. Find Optimal K (Elbow Method)
# ----------------------------
inertia_list = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia_list.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_range, inertia_list, marker='o', linestyle='--')
plt.xlabel('Number of clusters K')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal K')
plt.grid(True)
plt.savefig('outputs/elbow_method.png')
plt.close()
print("Elbow method plot saved to outputs/elbow_method.png")

# ----------------------------
# 5. Fit K-Means with optimal K
# ----------------------------
optimal_k = 5  # choose based on elbow
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# ----------------------------
# 6. Evaluate clustering using Silhouette Score
# ----------------------------
sil_score = silhouette_score(X_scaled, data['Cluster'])
print(f"Silhouette Score for K={optimal_k}: {sil_score:.4f}")

# ----------------------------
# 7. Visualize clusters in 2D using PCA
# ----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
data['PCA1'] = X_pca[:,0]
data['PCA2'] = X_pca[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(
    x='PCA1', y='PCA2',
    hue='Cluster',
    palette='Set1',
    data=data,
    legend='full',
    s=80
)
plt.title(f'K-Means Clusters (K={optimal_k})')
plt.savefig('outputs/kmeans_clusters.png')
plt.close()
print("2D cluster plot saved to outputs/kmeans_clusters.png")

# ----------------------------
# 8. Show first few rows with cluster labels
# ----------------------------
print(data.head())
