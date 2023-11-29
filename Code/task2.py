import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re

import torchvision
from torchvision import datasets
import torch

import os

# parse string
def parse_string(string):
    values = re.findall(r'-?\d+\.\d+', string)
    np_array = np.array(values, dtype=float)
    return np_array

###############################################################
# Cluster Helper Functions
###############################################################
# similarity/distance function
def euclidean_dist(p1, p2):
    dist = np.sqrt(np.sum((p1 - p2) ** 2))
    return dist
    
# min max normalization
def min_max_normalize(data):
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)
    return (data-min)/(max-min)

# z score normalization
def z_score_normalize(data):
    mean_val = np.mean(data)
    std_dev = np.std(data)
    return (data - mean_val) / std_dev

# find neighbors of a point
def find_neighbors(data, point, eps):
    neighbors = []
    for i, x in enumerate(data):
        # if the distance is within eps, add to neighbors
        if euclidean_dist(point, x) <= eps:
            neighbors.append(i)
            
    return neighbors

def DBSCAN(data):
    # calculate appropriate eps
    total_dist = 0
    count = 0
    for i in range(int(len(data))):
        for j in range(i+1, int(len(data))):
            distance = np.linalg.norm(data[i]-data[j])
            total_dist += distance
            count +=1

   
    avg_dist = total_dist / count if count > 0 else 0
    # print(avg_dist)

    eps = avg_dist * 0.9
    if(len(data) < 25): minPts = 1
    elif(len(data) < 100): minPts = 2
    elif(len(data) < 300): minPts = 3
    else: minPts = 5
        
    # create labels for each point
    labels = [-1] * len(data)

    # cluster counter
    cluster = 0

    for point in range(len(data)):
        # skip if already visited
        if labels[point] != -1:
            continue

        # find neighbors
        neighbor_pts = find_neighbors(data, data[point], eps)
    
        # label as noise if number of neighbors is less than minPts
        if len(neighbor_pts) < minPts:
            labels[point] = 0
            continue

        # start new cluster
        cluster += 1
        labels[point] = cluster

        # process every point in the neighborhood
        i = 0
        while i < len(neighbor_pts):
            pn = neighbor_pts[i]

            # border point
            if labels[pn] == 0:
                labels[pn] = cluster

            # new point
            elif labels[pn] == -1:
                labels[pn] = cluster
                pn_neighbor_pts = find_neighbors(data, data[pn], eps)
                if len(pn_neighbor_pts) >= minPts:
                    neighbor_pts += pn_neighbor_pts

            # go to next point
            i += 1

    #return 
    return labels

# top clusters
def top_clusters(labels, num_clusters):
    unique, counts = np.unique(labels, return_counts=True)
    cluster_counts = dict(zip(unique, counts))

    cluster_counts.pop(0,None)
    
    largest_clusters = sorted(cluster_counts, key=cluster_counts.get, reverse=True)[:num_clusters]
    return largest_clusters

###############################################################
# Visualization Helper Functions
###############################################################
# PCA dimensionality reduction
def PCA(data, num_components):
    # calculate covariance matrix
    covar_matrix = np.cov(data, rowvar=False)

    # calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covar_matrix)

    # sort in descending order
    sorted = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted]
    eigenvectors = eigenvectors[:, sorted]

    # choose top number of components
    top_eigenvectors = eigenvectors[:, :num_components]

    # project original data onto eigenvectors
    pca_data = np.dot(data, top_eigenvectors)

    # calculate amount of variance explained by each principal component
    total_var = np.sum(eigenvalues)
    explained_var = eigenvalues[:num_components] / total_var

    return pca_data

# visualize the clusters as differently colored point clouds in a 2-dimensional MDS space
def visualize_clouds(label, data, clusters, num_clusters):

    # PCA dimensionality reduction
    data_2D = PCA(data, 2)

    # plot different colored point clouds
    plt.figure(figsize=(10,8))
    unique_clusters = np.unique(clusters)
    
    for cluster in unique_clusters:
        if cluster == 0:
            continue
        if cluster > num_clusters: 
            continue
        cluster_indices = np.where(clusters == cluster)
        plt.scatter(data_2D[cluster_indices, 0], data_2D[cluster_indices, 1], label = f'Cluster {cluster}')
     
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'Label {label} Clusters in 2D PCA Space')
    plt.legend()

    folder_path = "t2_output\point_clusters"
    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


    file_name = "t2_label" + str(label)  + ".png" 
    file_path = os.path.join(folder_path, file_name)

    plt.savefig(file_path, dpi=150)
    plt.close()

# visualize as groups of thumbnails
def visualize_images(label, data, label_data, clusters, significant_clusters):

    folder_path = "t2_output\image_clusters"
    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # for each cluster group
    for cluster in significant_clusters:
        
        # get indices of that cluster group
        cluster_indices = np.where(clusters == cluster)

        # calculate number of rows and columsn based on number of images
        total_images = len(cluster_indices[0])
        num_cols = int(np.ceil(np.sqrt(total_images)))
        num_rows = int(np.ceil(total_images / num_cols))

        # create a figure and a set of subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
        axes = np.ravel(axes)
   
        plt.suptitle(f'Label: {label}      Cluster: {cluster}')
        plt.tight_layout()

        for i in range(len(axes)):
            axes[i].axis('off')
            
        for i, index in enumerate(cluster_indices[0]):
            # get imageID number for cluster point
            image_id = label_data.iloc[index, 0]
            # get image data
            image = data[image_id][0]

            # update subplot
            try:
                axes[i].imshow(image)
            except IndexError:
                break
        
        file_name = "t2_label" + str(label) + "_c" + str(i) + ".png" 
        file_path = os.path.join(folder_path, file_name)

        plt.savefig(file_path, dpi=150)
        plt.close()
###############################################################
# Prediction Helper Functions
###############################################################
def calculate_centroid(points):
    return np.mean(points, axis=0)

def find_nearest_cluster(img, cluster_centroids):
    nearest_label = None
    min_dist = float('inf')
    
    for label, centroid in cluster_centroids.items():
        distance = euclidean_dist(img, centroid)
        if distance < min_dist:
            min_dist = distance
            nearest_label = label
            
    return nearest_label


###############################################################
# Score Helper Function
###############################################################
# per-label precision, recall, and F1-score values
def calc_metrics(true_labels, predicted_labels):
    precision = {}
    recall = {}
    f1_score = {}

    #calculate precision, recall, F1 for each label
    labels = set(true_labels)
    
    for label in labels:
        #true positive
        tp = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred == label)
        #false positive
        fp = sum(1 for pred in predicted_labels if pred == label and pred not in true_labels)
        #false negative
        fn = sum(1 for true in true_labels if true == label and true not in predicted_labels)

        precision[label] = tp / (tp+fp) if tp+fp != 0 else 0
        recall[label] = tp / (tp+fn) if tp+fn != 0 else 0
        f1_score[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label]) if precision[label]+recall[label] != 0 else 0

    # accuracy
    accuracy = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred) / len(true_labels)

    return precision, recall, f1_score, accuracy



###############################################################
# Find Clusters
###############################################################

# get user input
num_clusters = int(input('Please enter a value for c: '))

# get features
df = pd.read_csv('FD_Objects_all.csv')
df_even = df[df['ImageID'] % 2 == 0]
df_odd = df[df['ImageID'] % 2 != 0]

# dataset
dataset = torchvision.datasets.Caltech101('data')

# set up output folder
folder_path = "t2_output"
# Check if the folder exists, if not, create it
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
        
results = {}
cluster_centroids = {}

for label in df_even['Labels'].unique():
    # filter rows with current label
    label_data = df_even[df_even['Labels'] == label]

    # get the list of features
    features = np.array(label_data['ResNet_FC_1000'].map(parse_string).tolist())
    #normalize feature data
    normalized_features = min_max_normalize(features)
    
    # run DBSCAN algorithm
    cluster_labels = DBSCAN(normalized_features)

    # find significant clusters
    significant_clusters = top_clusters(cluster_labels, num_clusters)
    results[label] = significant_clusters

    # compute cluser centroids for prediction part
    for cluster in significant_clusters:
        cluster_indices = np.where(cluster_labels == cluster)
    
        cluster_pts = normalized_features[cluster_indices]              
        cluster_centroids[(label,cluster)] = calculate_centroid(cluster_pts)
        
    #visualize 
    print('Label', label)
    print('c most significant clusters', significant_clusters)
    # print('num points', (len(features)))
    visualize_clouds(label, normalized_features, cluster_labels, num_clusters)
    visualize_images(label, dataset, label_data, cluster_labels, significant_clusters)

###############################################################
# Predict Likely Labels
###############################################################

df_odd = df[df['ImageID'] % 2 != 0]
features_odd = np.array(df_odd['ResNet_FC_1000'].map(parse_string).tolist())
normalized_features_odd = min_max_normalize(features_odd)

# predict label for each odd image
predicted_labels = []
for feature in normalized_features_odd:
    predicted_label = find_nearest_cluster(feature, cluster_centroids)
    predicted_labels.append(predicted_label[0])

# print to file
file_name = 't2_label_predictions.txt'
file_path = os.path.join(folder_path, file_name)

with open(file_path, 'w') as file:
    for i in range(len(predicted_labels)):
        content = f'Image ID: {i*2 +1} \n Predicted Label: {predicted_labels[i]} \n\n'
        file.write(content)


###############################################################
# Evaluate
###############################################################

# get true labels
true_labels = df_odd['Labels']

precision, recall, f1, accuracy = calc_metrics(true_labels, predicted_labels)

# print results to file
file_name = 't2_prediction_scores.txt'
file_path = os.path.join(folder_path, file_name)

with open(file_path, 'w') as file:
    file.write("Per-label Metrics:\n\n")
    for label in set(true_labels):
        content = f"Label: {label}, Precision: {precision[label]}, Recall: {recall[label]}, F1-Score: {f1[label]}\n\n"
        file.write(content)

    file.write(f"Overall Accuracy: {accuracy}")

