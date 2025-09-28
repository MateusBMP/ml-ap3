from sklearn.cluster import DBSCAN, KMeans

import common
import numpy as np
import pandas as pd

# Load dataset using pandas
df = pd.read_excel("barrettII_eyes_clustering.xlsx")

# Remove the column ID
df = df.drop(columns=["ID"])

# Remove the column Correto
df = df.drop(columns=["Correto"])

# Normalize the dataset
df = (df - df.min()) / (df.max() - df.min())

# Some analysis on the dataset
common.dataset_analysis(df)

# Apply DBSCAN clustering algorithm
clustering = DBSCAN(eps=0.075, min_samples=5).fit(df)
df['Cluster'] = clustering.labels_

# Print clustering results
print("##### Clustering Results #####")
print("Cluster labels assigned to each data point:")
print(df['Cluster'].value_counts())
print()
for cluster in np.unique(clustering.labels_):
    sub_df = df[df['Cluster'] == cluster]
    print(sub_df.head())

# Creating decision tree to understand clusters
common.plot_decision_tree(df, filename="dt_v1_1.png")

# Plot scatter plot for all pairs of features
common.plot_scatter_matrix(df, filename="sc_v1_1.png")

# Print distances between clusters
common.distance(df, output=True)

# Apply KMeans clustering algorithm only in the data points that were classified as noise or assigned to cluster 0
noise_df = df[df['Cluster'] == -1]
cluster_0 = df[df['Cluster'] == 0]
combined_df = pd.concat([noise_df, cluster_0])
kmeans = KMeans(n_clusters=8, random_state=42).fit(combined_df.drop(columns=['Cluster']))
df.loc[combined_df.index, 'Cluster'] = kmeans.labels_ + df['Cluster'].max() + 1

# Print new clustering results
print("##### New Clustering Results after KMeans #####")
print("Cluster labels assigned to each data point:")
print(df['Cluster'].value_counts())
print()
for cluster in np.unique(df['Cluster']):
    sub_df = df[df['Cluster'] == cluster]
    print(sub_df.head())
    print()

# Creating, again, decision tree to understand clusters
common.plot_decision_tree(df, filename="dt_v1_2.png")

# Plot scatter plot for all pairs of features
common.plot_scatter_matrix(df, filename="sc_v1_2.png")

# Print distances between clusters
common.distance(df, output=True)
