import os
import re
from datetime import datetime
import csv
import PIL
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from numpy.random import randn
from torchvision import models, datasets
from torchvision.transforms import transforms

torch.set_grad_enabled(False)

ROOT_DIR = '/home/rpaw/MWD/caltech-101/caltech-101/101_ObjectCategories/'
CNN_MODEL = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
BASE_DIR = '/home/rpaw/MWD/'


# Your parse_string function
def parse_string(string):
    values = re.findall(r'-?\d+\.\d+', string)
    np_array = np.array(values, dtype=float)
    return torch.tensor(np_array, dtype=torch.float32)

# Function to plot k similar images against input image for all 5 feature models
def plot_result(feature_vector_similarity_sorted_pairs, t, input_image_id_or_path, layers, hashes):
    dataset = datasets.Caltech101(BASE_DIR)
    # Number of images per row
    images_per_row = t
    # Number of rows needed(1 Original image + 1 LSH results)
    num_rows = 2
    fig, axes = plt.subplots(num_rows, images_per_row + 1, figsize=(30, 10))
    plt.subplots_adjust(wspace=0.5)

    # Load and display the original image
    original_label = f"Input Image:"
    original_img = get_image(input_image_id_or_path)

    axes[0, 1].imshow(original_img, cmap="gray")
    axes[0, 1].axis('off')
    axes[0, 0].set_title(original_label, loc='left', pad=10, verticalalignment='center')
    axes[0, 0].axis('off')

    for i in range(1):
        axes[i + 1, 0].set_title(f"{t} similar images using LSH with {layers} layer(s) and {hashes} hash(es)",x=1,
                                 loc='center', pad=10,
                                 verticalalignment='top')
        axes[i + 1, 0].axis('off')
        for j in range(images_per_row):
            if j < len(feature_vector_similarity_sorted_pairs):
                image_id_index, similarity_score = feature_vector_similarity_sorted_pairs[j]
                img = dataset[image_id_index][0]
                axes[i + 1, j + 1].imshow(img, cmap="gray")
                axes[i + 1, j + 1].set_title(
                    f'Euclidian Distance: {similarity_score:.2f}', pad=5,
                    verticalalignment='top')
                axes[i + 1, j + 1].axis('off')

    # Removed empty subplot in row 0
    for j in range(0, images_per_row + 1):
        if j in [0, 1]:
            continue
        fig.delaxes(axes[0, j])

    plt.tight_layout()

    # Saved output to output dir
    current_epoch_timestamp = int(datetime.now().timestamp())
    output_path = f"../Outputs/T4/id_{input_image_id_or_path}_t_{t}_layers_{layers}_hashes_{hashes}_ts_{current_epoch_timestamp}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()


def get_dataset_feature_set():
    # Read the features from the CSV file using pandas
    df = pd.read_csv('./../FD_Objects_all.csv')

    # Apply the parse_string function to the desired column
    df['ResNet_FC_1000'] = df['ResNet_FC_1000'].apply(parse_string)

    # Split the data into even and odd subsets
    image_superset_features = df.iloc[::2]
    return image_superset_features


def bool_to_int(x):
    res = 0
    for val in x:
        if val:
            res += 1
        res = res << 1
    return res


def hash_func(vecs, projections):
    bools = np.dot(vecs, projections.T) > 0
    return bool_to_int(bools)


# Function to calculate euclidian distance between 2 vectors
def calculate_euclidian_distance(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    return np.linalg.norm(vector1 - vector2)


# Generic function to extract output of intermediate layers of Resnet 50 model
def extract_feature_vector(image, layer):
    hook_output = []

    def hook_fn(module, input, output):
        hook_output.append(output)

    # Attached a hook to the input layer
    hook_layer = layer.register_forward_hook(hook_fn)

    # Loaded and preprocessed image
    image = preprocess_image(image)

    CNN_MODEL.eval()

    # Forward Passed the image through the model
    with torch.no_grad():
        CNN_MODEL(image)

    hook_layer.remove()

    return hook_output[0].squeeze()


# Function to preprocess image before feeding to Resnet50 feature extractor
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    transformed_image = transform(image).unsqueeze(0)
    return transformed_image


# Function to extract Resnet 50 Fully Connected layer features
def extract_resnet_fc_1000(image):
    layer = CNN_MODEL.fc
    feature_output = extract_feature_vector(image, layer)
    return feature_output.numpy()
    
# Writing the data to the CSV file
def save_output_csv(data):
    csv_file_path = '../Outputs/T4/task4.csv'

    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        csv_writer.writerow(['Image ID', 'Euclidean Distance'])
        
        csv_writer.writerows(data)
        
class Layer:

    def __init__(self, hash_size, dim):
        self.table = dict()
        self.hash_size = hash_size
        # random projections(hyperplane)
        self.projections = randn(self.hash_size, dim)

    def add(self, vecs, label):
        entry = {'label': label}
        hashcode = hash_func(vecs, self.projections)
        if hashcode in self.table.keys():
            self.table[hashcode].append(entry)
        else:
            self.table[hashcode] = [entry]

    def query(self, vecs):
        hashcode = hash_func(vecs, self.projections)
        results = list()
        if hashcode in self.table.keys():
            results.extend(self.table[hashcode])
        return results


class LSH:

    def __init__(self, dim, num_tables, hash_size):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.tables = list()
        for i in range(self.num_tables):
            self.tables.append(Layer(self.hash_size, dim))

    def add(self, vecs, label):
        for table in self.tables:
            table.add(vecs, label)

    def query(self, vecs):
        results = list()
        for table in self.tables:
            results.extend(table.query(vecs))
        return results

    def describe(self):
        for table in self.tables:
            print(table.table)


class ApproximateNearestNeighborSearch:

    def __init__(self, layers, hashes):
        self.layers = layers
        self.hashes = hashes
        self.image_vector_dimension = 1000
        self.lsh = LSH(self.image_vector_dimension, self.layers, self.hashes)

    def train(self):
        feature_set = get_dataset_feature_set()
        for i in range(len(feature_set)):
            feature = feature_set.iloc[i]
            self.lsh.add(np.array(feature["ResNet_FC_1000"]), feature["ImageID"])

    def find_t_nearest_neighbor(self, input_image_id_or_path, t, approach=1):
        query_image_vector = get_input_image_vector(input_image_id_or_path)
        result_set = self.query(query_image_vector)
        even_features = get_dataset_feature_set()

        # approach 1: calculate euclidian of t that have the highest occurrences in all tables(less # of comparisons)
        if approach == 1:
            image_id_to_result_count_dict = {}
            for result in result_set:
                image_id = result["label"]
                if image_id not in image_id_to_result_count_dict.keys():
                    image_id_to_result_count_dict[image_id] = 0
                image_id_to_result_count_dict[image_id] += 1
            sorted_image_id_frequency_list = sorted(image_id_to_result_count_dict.items(), key=lambda item: item[1],
                                                    reverse=True)
            result_set = []
            if len(sorted_image_id_frequency_list) > 1:
                i = 0
                freq = 0
                for i in range(t):
                    result_set.append({"label": sorted_image_id_frequency_list[i][0]})
                    freq = sorted_image_id_frequency_list[i][1]
                i += 1
                while i < len(sorted_image_id_frequency_list) and sorted_image_id_frequency_list[i][1] == freq:
                    result_set.append({"label": sorted_image_id_frequency_list[i][0]})
                    i += 1

        # approach 2: calculate euclidian of l*h(compare result set as is)
        euclidian_distance_list = []
        for result in result_set:
            # for entry in table_result_set:
            image_id = result["label"]
            image_vector = even_features.iloc[image_id//2]["ResNet_FC_1000"]
            euclidian_distance = calculate_euclidian_distance(query_image_vector, image_vector)
            euclidian_distance_list.append((image_id, euclidian_distance))

        total_images_count = len(euclidian_distance_list)
        unique_images_count = get_unique_images_count(euclidian_distance_list)
        # unique t images
        euclidian_distance_list = list(set(euclidian_distance_list))
        euclidian_distance_list = sorted(euclidian_distance_list, key=lambda x: x[1])
        plot_result(euclidian_distance_list[:t], t, input_image_id_or_path, self.layers, self.hashes)
        print(f"Numbers of unique images considered during the process: {unique_images_count}")
        print(f"Overall number of images considered during the process: {total_images_count}")
        save_output_csv(euclidian_distance_list)

    def query(self, query_image_vector):
        return self.lsh.query(query_image_vector)

    def describe(self):
        self.lsh.describe()


def get_input_image_vector(input_image_id_or_path):
    even_features = get_dataset_feature_set()
    input_img = get_image(input_image_id_or_path)

    if input_image_id_or_path.isnumeric() and int(input_image_id_or_path) % 2 == 0:
        input_image_id = int(input_image_id_or_path)
        input_image_features = np.array(even_features.iloc[input_image_id//2]["ResNet_FC_1000"])
    else:
        input_image_features = extract_resnet_fc_1000(input_img)

    return input_image_features


def get_image(input_image_id_or_path):
    dataset = datasets.Caltech101(BASE_DIR)
    if input_image_id_or_path.isnumeric():
        input_image_id = int(input_image_id_or_path)
        input_img = dataset[input_image_id][0]
    else:
        input_img = PIL.Image.open(input_image_id_or_path)
    return input_img


def get_unique_images_count(euclidian_distance_list):
    unique_elements = set()
    for item in euclidian_distance_list:
        unique_elements.add(item[1])
    return len(unique_elements)


def driver():
    # l = int(input("Please select numer of Layers, L\n"))
    # h = int(input("Please number of hashes per layer, h\n"))
    # input_image_id_or_path = input("Please select an image id or image path\n")
    # t = int(input("Please select t to find t similar images\n"))
    l = 6
    h = 7
    t = 10
    input_image_id_or_path = "3100"
    ann = ApproximateNearestNeighborSearch(l, h)
    ann.train()
    ann.find_t_nearest_neighbor(input_image_id_or_path, t)
    ann.describe()


if __name__ == "__main__":
    driver()
