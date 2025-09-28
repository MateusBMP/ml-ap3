from sklearn.cluster import KMeans
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
pd.plotting.scatter_matrix(df, c=df['Cluster'], figsize=(15, 15), marker='o',
                           hist_kwds={'bins': 20}, s=60, alpha=.8, cmap='viridis')
plt.suptitle("Scatter Matrix of Features Colored by Cluster")
plt.savefig("sc_v2_1.png")

# Creating decision tree to understand clusters
X = df.drop(columns=["Cluster"])
y = df["Cluster"]
plt.figure(figsize=(200,100))
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)
tree.plot_tree(clf, filled=True)
plt.title("Decision Tree for Clusters")
plt.savefig("dt_v2_1.png")

# Algorithm to calculate the distance between clusters
def distance(df):
    means = df.groupby('Cluster').mean().values
    combinations = [(i, j) for i in range(len(means)) for j in range(i + 1, len(means))]
    distances = {}
    for (i, j) in combinations:
        dist = np.linalg.norm(means[i] - means[j])
        distances[(i, j)] = dist

    # print("##### Distances Between Clusters #####")
    # for (i, j), dist in distances.items():
    #     print(f"Distance between Cluster {i} and Cluster {j}: {dist:.4f}")
    # print("Total mean distance between clusters:", np.mean(list(distances.values())))
    # print()

    return np.mean(list(distances.values()))

# Run a combination of KMeans with different number of clusters
# results = {}
# for n_clusters in range(2, 20):
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(df.drop(columns=["Cluster"]))
#     df['Cluster'] = kmeans.labels_
#     results[n_clusters] = distance(df)

# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(list(results.keys()), list(results.values()), marker='o')
# plt.title("Mean Distance Between Clusters vs Number of Clusters")
# plt.xlabel("Number of Clusters")
# plt.ylabel("Mean Distance Between Clusters")
# plt.xticks(range(2, 20))
# plt.grid()
# plt.savefig("md_v2_1.png")