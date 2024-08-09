import pandas as pd
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def get_sorted_cluster_members(points, clusters, centroids):
    """
    For each cluster, return and print the signal names sorted by their distance to the cluster centroid.
    """
    cluster_members = {i: [] for i in range(len(centroids))}
    for i, cluster_id in enumerate(clusters):
        cluster_members[cluster_id].append((i, points[i]))

    sorted_clusters = {}
    for cluster_id, members in cluster_members.items():
        sorted_members = sorted(members, key=lambda x: np.linalg.norm(x[1] - centroids[cluster_id]))
        sorted_signal_names = [correlation_matrix.columns[x[0]] for x in sorted_members]
        sorted_clusters[cluster_id] = sorted_signal_names
        print(f"Cluster {cluster_id}: {sorted_signal_names}")

    return sorted_clusters

def find_optimal_clusters(points, max_k=10):
    """
    Use the Elbow method and Silhouette method to find the optimal number of clusters.
    """
    inertia = []
    silhouette_scores = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(points)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(points, kmeans.labels_))

    # Find the optimal number of clusters
    optimal_k_elbow = np.argmin(np.gradient(inertia)) + 2
    optimal_k_silhouette = np.argmax(silhouette_scores) + 2

    return optimal_k_elbow, optimal_k_silhouette

def correlation_to_distance(correlation_matrix):
    """
    Converts a correlation matrix to a distance matrix using the formula:
    distance(i, j) = sqrt(2 * (1 - correlation(i, j)))
    """
    return np.sqrt(2 * (1 - correlation_matrix))

if __name__ == '__main__':
    # Load the data
    data_csv = pd.read_csv('resources/PredictorPortsFull.csv')

    # Filter rows where 'port'(second column) is 'LS' and remove 'signallag' column
    data_ls = data_csv[data_csv['port'] == 'LS']
    data_ls = data_ls.drop(columns=['signallag'])
    data_ls = data_ls.drop(columns=['port'])
    print(data_ls)

    # Make Correlation Matrix
    # shows the correlation between each signalname
    # it show how returns (ret) are correlated with each other using date column

    # Pivot the data to have signalname as columns, date as index, and ret as values
    pivot_data = data_ls.pivot(index='date', columns='signalname', values='ret')
    correlation_matrix = pivot_data.corr()
    # print("Correlation Matrix:")
    # print(correlation_matrix)

    # Convert the correlation matrix to a distance matrix
    distance_matrix = correlation_to_distance(correlation_matrix)
    # print("Distance Matrix:")
    # print(distance_matrix)

    # Apply Multi-Dimensional Scaling (MDS) to reduce to 2D
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    feature_vectors = mds.fit_transform(distance_matrix)

    # Find the optimal number of clusters
    optimal_k_elbow, optimal_k_silhouette = find_optimal_clusters(feature_vectors, max_k=10)

    # Choose the optimal k (can be either optimal_k_elbow or optimal_k_silhouette)
    k = optimal_k_silhouette

    # Apply K-means clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(feature_vectors)
    centroids = kmeans.cluster_centers_

    # Get and print sorted cluster members
    sorted_clusters = get_sorted_cluster_members(feature_vectors, clusters, centroids)

    # Plot the 2D points with cluster assignments
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(feature_vectors[:, 0], feature_vectors[:, 1], c=clusters, cmap='viridis')

    # Annotate points with signal names
    for i, signalname in enumerate(correlation_matrix.columns):
        plt.annotate(signalname, (feature_vectors[i, 0], feature_vectors[i, 1]))

    plt.title(f'2D representation of signal names using MDS with K-means Clustering (k={k})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.colorbar(scatter, label='Cluster')
    plt.show()