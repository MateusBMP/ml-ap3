from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn import tree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load dataset using pandas
df = pd.read_excel("barrettII_eyes_clustering.xlsx")

# Remove the column ID
df = df.drop(columns=["ID"])

# Remove the column Correto
df = df.drop(columns=["Correto"])

# Some analysis on the dataset
print("##### Dataset Analysis #####")
print("Dataset shape:", df.shape)
print("First 5 rows of the dataset:")
print(df.head())
print("Column names:", df.columns.tolist())
print("Missing values in each column:")
print(df.isnull().sum())
print("Statistical summary of the dataset:")
print(df.describe())
print("Data types of each column:")
print(df.dtypes)
print()

# Apply DBSCAN clustering algorithm
clustering = DBSCAN(eps=0.5, min_samples=5).fit(df)
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
X = df.drop(columns=["Cluster"])
y = df["Cluster"]
plt.figure(figsize=(200,100))
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)
tree.plot_tree(clf, filled=True)
plt.title("Decision Tree for Clusters")
plt.savefig("decision_tree.png")

# Apply KMeans clustering algorithm only in the data points that were classified as noise or assigned to cluster 0
noise_df = df[df['Cluster'] == -1].drop(columns=["Cluster"])
cluster_0 = df[df['Cluster'] == 0].drop(columns=["Cluster"])
combined_df = pd.concat([noise_df, cluster_0])
kmeans = KMeans(n_clusters=3, random_state=42).fit(combined_df)
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
X = df.drop(columns=["Cluster"])
y = df["Cluster"]
plt.figure(figsize=(200,100))
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)
tree.plot_tree(clf, filled=True)
plt.title("Decision Tree for Clusters")
plt.savefig("decision_tree_v2.png")