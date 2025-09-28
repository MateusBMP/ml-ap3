from sklearn.cluster import DBSCAN
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

# Normalize the dataset
df = (df - df.min()) / (df.max() - df.min())

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
clustering = DBSCAN(eps=0.075, min_samples=5).fit(df)
df['Cluster'] = clustering.labels_

# Print new clustering results
print("##### KMeans Clustering Results #####")
print("Cluster labels assigned to each data point:")
print(df['Cluster'].value_counts())
print()
for cluster in np.unique(clustering.labels_):
    sub_df = df[df['Cluster'] == cluster]
    # print(sub_df.head())
    print(sub_df.describe())
    print("#########################################")

# Plot scatter plot for all pairs of features
pd.plotting.scatter_matrix(df, c=df['Cluster'], figsize=(15, 15), marker='o',
                           hist_kwds={'bins': 20}, s=60, alpha=.8, cmap='viridis')
plt.suptitle("Scatter Matrix of Features Colored by Cluster")
plt.savefig("sc_v3_1.png")

# Creating decision tree to understand clusters
X = df.drop(columns=["Cluster"])
y = df["Cluster"]
plt.figure(figsize=(200,100))
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)
tree.plot_tree(clf, filled=True)
plt.title("Decision Tree for Clusters")
plt.savefig("dt_v3_1.png")