import common
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans

# Run a combination of KMeans with different number of clusters
def optimize_kmeans(df, 
                    cluster_col: str = 'Cluster', 
                    start: int = 2, 
                    end: int = 20, 
                    plot: bool = True, 
                    filename: str = "kmeans_optimization.png"):
    results = {}
    
    for n_clusters in range(start, end):
        prepared_df = df.drop(columns=[cluster_col]) if cluster_col in df.columns else df
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(prepared_df)
        df[cluster_col] = kmeans.labels_
        results[n_clusters] = common.distance(df)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(list(results.keys()), list(results.values()), marker='o')
        plt.title("Mean Distance Between Clusters vs Number of Clusters")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Mean Distance Between Clusters")
        plt.xticks(range(start, end))
        plt.grid()
        plt.savefig(filename)

    return results
