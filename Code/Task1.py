#!/usr/bin/env python
# coding: utf-8

# In[154]:


# Task 1
# Add all the required libraries here, will be using dimensionality reduction technique of SVD
import torch
import numpy as np
import pandas as pd
import re


# In[155]:


# Adding all the required functions here
# parse_string function
def parse_string(string):
    values = re.findall(r'-?\d+\.\d+', string)
    np_array = np.array(values, dtype=float)
    return torch.tensor(np_array, dtype=torch.float32)

# this function is to perform SVD
def svd(k, feature_matrix):
    covariance_matrix_1 = np.dot(feature_matrix.T, feature_matrix)
    eigenvalues_1, eigenvectors_1 = np.linalg.eig(covariance_matrix_1)
    ncols1 = np.argsort(eigenvalues_1)[::-1]
    covariance_matrix_2 = np.dot(feature_matrix, feature_matrix.T)
    eigenvalues_2, eigenvectors_2 = np.linalg.eig(covariance_matrix_2)
    ncols2 = np.argsort(eigenvalues_2)[::-1]
    v_transpose = eigenvectors_1[ncols1].T
    u = eigenvectors_2[ncols2]
    sigma = np.diag(np.sqrt(eigenvalues_1)[::-1])
    trucated_u = u[:, :k]
    trucated_sigma = sigma[:k, :k]
    truncated_v_transpose = v_transpose[:k, :]
    image_to_latent_features = feature_matrix @ truncated_v_transpose.T
    latent_feature_to_original_feature = truncated_v_transpose
    # svd = TruncatedSVD(n_components=k)
    # reduced_data = svd.fit_transform(feature_matrix)
    # image_to_latent_features = feature_matrix @ v_transpose.T
    # latent_feature_to_original_feature = v_transpose
    return image_to_latent_features, latent_feature_to_original_feature

#  for each label, I am going to do label semantic analysis, one by one, the value of k=10
def perform_svd(k, data, labels):
    # first thing I need to do is convert everything to a numpy array so that I can pass to the SVD function
    reduced_features = {}
    for label in labels:
        label_data = data.get_group(label)[feature_model]
        # Convert the list of arrays to a 2D NumPy array
        list_data = np.vstack([np.array(image) for image in label_data])

        # Performing Singular Value Decomposition (SVD)
        image_to_latent_features, _ = svd(k, list_data)

        # Storing the reduced features in the dictionary
        reduced_features[label] = image_to_latent_features
    return reduced_features

def cosine_similarity(vector_a, vector_b):
    # Calculated the dot product of the two vectors
    dot_product = np.dot(vector_a, vector_b)

    # Calculated the Euclidean norm (magnitude) of each vector
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    # Calculated the cosine similarity
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def calculate_similarity(latent_semantics1, latent_semantics2):
    """Calculate the cosine similarity between two latent semantics."""
    return cosine_similarity(latent_semantics1, latent_semantics2)

def calculate_total_accuracy(predicted_labels):
    total_num = 0
    true_positives = 0
    for label in predicted_labels.keys():
        for image_label in predicted_labels[label]:
            total_num += 1
            if image_label == label:
                true_positives += 1
    return true_positives/total_num

# function to calculate per-label precision, recall, and F1-score
def calculate_metrics(predicted_labels):
    num_labels = len(predicted_labels)
    label_metrics = {}
    for label in predicted_labels.keys():
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for image_label in predicted_labels[label]:
            if image_label == label:
                true_positives += 1
            else:
                false_positives += 1
        false_negatives = num_labels - true_positives  # All instances not considered as true positives

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        label_metrics[label] = [precision, recall, f1_score]
    return label_metrics

def print_metrics(label_metrics, predicted_labels):
    for label in label_metrics:
        print(f"Label {label}: Precision: {label_metrics[label][0]}, Recall: {label_metrics[label][1]}, f1 Score: {label_metrics[label][2]}.")
    print(f"Total accuracy: {calculate_total_accuracy(predicted_labels)}")
    
def select_feature_model(option):
    if option == 1:
        return "HOG"
    elif option == 2:
        return "ColorMoments"
    elif option == 3:
        return "ResNet_AvgPool_1024"
    elif option == 4:
        return "ResNet_Layer3_1024"
    else:
        return "ResNet_FC_1000"
    


# In[156]:


# Step 1: Feature Extraction and Storage
# reading the feature file and using the resnet fc model for extraction
df = pd.read_csv('FD_Objects.csv')

feature_option = int(input("Please pick one of the below options\n"
                               "1. HOG\n"
                               "2. Color Moments\n"
                               "3. Resnet Layer 3\n"
                               "4. Resnet Avgpool\n"
                               "5. Resnet FC\n"
                               "--------------\n"))
k = int(input("Please enter the value of k: "))

feature_model = select_feature_model(feature_option)
df[feature_model] = df[feature_model].apply(parse_string)

# Convert the column to a NumPy array
numpy_array = df[feature_model].to_numpy()

# extracting the even and odd feature set 
even_data = df.iloc[::2]
odd_data = df.iloc[1::2]

# extracting labels
grouped_data_even = even_data.groupby('Labels')
labels = list(grouped_data_even.groups.keys())


# In[157]:


# Step 2: Latent Semantic Analysis (LSA)

# getting reduced data for even images in the label
reduced_feature_even = perform_svd(k, grouped_data_even, labels)

# getting reduced data for odd images in the label
grouped_data_odd = odd_data.groupby('Labels')
reduced_feature_odd = perform_svd(k, grouped_data_odd, labels)


# In[158]:


#  now we need to compare the similarity measure of each of the odd data with the even data images
predicted_label = {}

for label_odd in reduced_feature_odd.keys():
    predicted_label[label_odd] = []
    #comparing each odd image with even image
    for i in range(len(reduced_feature_odd[label_odd])):
        min_distance = np.zeros(5)
        label_min_distance = np.zeros(5)
        for label_even in reduced_feature_even.keys():
            for j in range(len(reduced_feature_even[label_even])):
                # now you can compare each odd image with each even image
                similarity = calculate_similarity(reduced_feature_odd[label_odd][i] , reduced_feature_even[label_even][j])
                #see if the similarity is greater than the values currently stored
                if similarity > np.max(min_distance):
#                     print("coming here", similarity, np.max(min_distance), min_distance)
                    min_index = np.argmin(min_distance)
                    min_distance[min_index] = similarity
                    label_min_distance[min_index] = label_even
                
                    
#                 print(similarity)
#                 print(label_odd, i,label_even, j, similarity)
                
#             break
#         max_index = np.where(min_distance == max(min_distance))
        most_frequent_value_np = np.bincount(label_min_distance.astype(int)).argmax()
        
#         print(i,most_frequent_value_np)
        predicted_label[label_odd].append(most_frequent_value_np)
#         break
#     break

        


# In[159]:


label_metrics = calculate_metrics(predicted_label)
print_metrics(label_metrics, predicted_label)

