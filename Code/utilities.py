import os
from datetime import datetime

import PIL
import numpy as np
import torch
from matplotlib import pyplot as plt
from pymongo import MongoClient
from torchvision import models, datasets
from torchvision.transforms import transforms

torch.set_grad_enabled(False)

ROOT_DIR = '/home/rpaw/MWD/caltech-101/caltech-101/101_ObjectCategories/'
CNN_MODEL = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
BASE_DIR = '/home/rpaw/MWD/'

# MongoDB Client Setup
MONGO_CLIENT = MongoClient("mongodb://adminUser:adminPassword@localhost:27017/mwd_db?authSource=admin")
DATABASE = MONGO_CLIENT['mwd_db']


# for actual union
def list_intersection(lists):
    if not lists:
        return []

    intersection_set = set((item["label"] for item in lists[0]))

    for l in lists[1:]:
        intersection_set.intersection_update(item["label"] for item in l)

    result = [{"label": label} for label in intersection_set]
    return result


# for finding neighbors
def generate_one_bit_diff_numbers(number, n):
    binary_representation = format(number, f'0{n}b')  # Convert the number to binary with n bits

    result = []
    for i in range(n):
        # Flip the bit at position i
        new_number = number ^ (1 << i)
        result.append(new_number)

    return result


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
        axes[i + 1, 0].set_title(f"{t} similar images using LSH with {layers} layer(s) and {hashes} hash(es)", x=1,
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
    dataset = datasets.Caltech101(BASE_DIR)
    collection = DATABASE.feature_descriptors
    image_superset_features = collection.find({}, {"resnet_fc_1000": 1, "image_id": 1, "_id": 0})
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




def get_input_image_vector(input_image_id_or_path):
    collection = DATABASE.feature_descriptors
    input_img = get_image(input_image_id_or_path)

    if input_image_id_or_path.isnumeric() and int(input_image_id_or_path) % 2 == 0:
        input_image_id = int(input_image_id_or_path)
        input_image_features = collection.find_one({"image_id": int(input_image_id)})["resnet_fc_1000"]
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
    input_img = input_img.convert('RGB')
    return input_img


def get_unique_images_count(euclidian_distance_list):
    unique_elements = set()
    for item in euclidian_distance_list:
        unique_elements.add(item[1])
    return len(unique_elements)