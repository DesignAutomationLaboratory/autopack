from sklearn.cluster import KMeans
import numpy as np

def select_KMeans(df, i):
    # Initialize KMeans
    kmeans = KMeans(n_clusters=i, random_state=0)

    # Fit KMeans on the DataFrame
    kmeans.fit(df)

    # Get the cluster labels
    labels = kmeans.labels_

    # Get unique labels and their counts
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Initialize an empty list to store selected designs
    selected_designs = []

    # Select one design from each cluster
    for label in unique_labels:
        # Get the indices of the designs in the current cluster
        design_indices = np.where(labels == label)[0]

        # Select one design randomly from the current cluster
        selected_design = np.random.choice(design_indices)

        # Append the selected design to the list
        selected_designs.append(selected_design)

    return selected_designs

