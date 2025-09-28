from sklearn import tree

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Plot scatter plot for all pairs of features
def plot_scatter_matrix(df: pd.DataFrame, cluster_col: str = 'Cluster', filename: str = "scatter_matrix.png"):
    pd.plotting.scatter_matrix(df, c=df[cluster_col], figsize=(15, 15), marker='o',
                            hist_kwds={'bins': 20}, s=60, alpha=.8, cmap='viridis')
    plt.suptitle("Scatter Matrix of Features Colored by Cluster")
    plt.savefig(filename)

# Creating decision tree to understand clusters
def plot_decision_tree(df: pd.DataFrame, cluster_col: str = 'Cluster', filename: str = "decision_tree.png"):
    X = df.drop(columns=[cluster_col])
    y = df[cluster_col]
    plt.figure(figsize=(200,100))
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)
    tree.plot_tree(clf, filled=True)
    plt.title("Decision Tree for Clusters")
    plt.savefig(filename)

def dataset_analysis(df: pd.DataFrame):
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

# Algorithm to calculate the distance between clusters
def distance(df: pd.DataFrame, group_by: str = 'Cluster', output: bool = False):
    means = df.groupby(group_by).mean().values
    combinations = [(i, j) for i in range(len(means)) for j in range(i + 1, len(means))]
    distances = {}
    for (i, j) in combinations:
        dist = np.linalg.norm(means[i] - means[j])
        distances[(i, j)] = dist

    total_mean_distance = np.mean(list(distances.values()))

    if output:
        print("##### Distances Between Clusters #####")
        for (i, j), dist in distances.items():
            print(f"Distance between Cluster {i} and Cluster {j}: {dist:.4f}")
        print("Total mean distance between clusters:", total_mean_distance)
        print()

    return total_mean_distance