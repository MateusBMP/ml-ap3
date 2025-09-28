from sklearn.cluster import KMeans

import common
import numpy as np
import optimization
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

# Apply Kmeans clustering algorithm
kmeans = KMeans(n_clusters=8, random_state=42).fit(df)
df['Cluster'] = kmeans.labels_

# Print new clustering results
print("##### KMeans Clustering Results #####")
print("Cluster labels assigned to each data point:")
print(df['Cluster'].value_counts())
print()
for cluster in np.unique(kmeans.labels_):
    sub_df = df[df['Cluster'] == cluster]
    # print(sub_df.head())
    print(sub_df.describe())
    print("##################################################################################")

# Plot scatter plot for all pairs of features
common.plot_scatter_matrix(df, filename="sc_v2_1.png")

# Creating decision tree to understand clusters
common.plot_decision_tree(df, filename="dt_v2_1.png")

# Print distances between clusters
common.distance(df, output=True)

# Optimize KMeans by varying the number of clusters and plotting the mean distance between clusters
optimization.optimize_kmeans(df, filename="md_v2_1.png")
