import os
import pickle

from numpy.random import randn
from torchvision import datasets

from utilities import hash_func, generate_one_bit_diff_numbers, get_dataset_feature_set, get_input_image_vector, \
    BASE_DIR, plot_result, result_set_processing


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

    def extended_query(self, vecs):
        hashcode = hash_func(vecs, self.projections)
        neighbours = generate_one_bit_diff_numbers(hashcode, self.hash_size)
        neighbours.append(hashcode)
        results = list()
        for hashcode in neighbours:
            if hashcode in self.table.keys():
                results.extend(self.table[hashcode])
        return results

    def query(self, vecs):
        hashcode = hash_func(vecs, self.projections)
        # print(hashcode)
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
            results.append(table.query(vecs))
        return results

    def extended_query(self, vecs):
        results = list()
        for table in self.tables:
            results.append(table.extended_query(vecs))
        return results

    def describe(self):
        for table in self.tables:
            print(table.table)
            print(table.table.keys())
            print([len(row) for row in table.table.values()])


def get_index_details():
    lsh_index_file_path = f'./LSH_INDICES/lsh_index_details.pkl'
    with open(lsh_index_file_path, 'rb') as file:
        lsh_index_details = pickle.load(file)
        print(f"LSH Index loaded")
    return lsh_index_details


# ApproximateNearestNeighborSearch Class to encapsulate training and nearest neighbor search
class ApproximateNearestNeighborSearch:

    def __init__(self, layers=None, hashes=None):
        self.image_vector_dimension = 1000
        if layers is None and hashes is None:
            lsh_index_details = get_index_details()
            self.layers = lsh_index_details["layers"]
            self.hashes = lsh_index_details["hashes"]
            self.lsh = lsh_index_details["lsh_index"]
        else:
            self.layers = layers
            self.hashes = hashes
            self.lsh = LSH(self.image_vector_dimension, self.layers, self.hashes)

    # Creates LSH index based on param layers and hashes
    def train(self):
        feature_set = get_dataset_feature_set()
        for feature in feature_set:
            self.lsh.add(feature["resnet_fc_1000"], feature["image_id"])
        lsh_index = self.lsh
        lsh_index_file_path = f'./LSH_INDICES/lsh_index_details.pkl'
        # Ensure that the directory path exists, creating it if necessary
        os.makedirs(os.path.dirname(lsh_index_file_path), exist_ok=True)
        with open(lsh_index_file_path, 'wb') as file:
            pickle.dump({"layers": self.layers, "hashes": self.hashes, "lsh_index": lsh_index}, file)
            print(f"LSH Index saved to {lsh_index_file_path}")

    # Searches t nearest neighbor to given image id or path
    def find_t_nearest_neighbor(self, input_image_id_or_path, t):
        query_image_vector = get_input_image_vector(input_image_id_or_path)
        dataset = datasets.Caltech101(BASE_DIR)
        result_set = self.query(query_image_vector)
        unique_images_count, total_images_count, euclidian_distance_list = result_set_processing(query_image_vector,
                                                                                                 result_set)
        if unique_images_count < t:
            print("Using extended query")
            result_set = self.extended_query(query_image_vector)
            unique_images_count, total_images_count, euclidian_distance_list = result_set_processing(query_image_vector,
                                                                                                     result_set)

        plot_result(euclidian_distance_list[:t], t, input_image_id_or_path, self.layers, self.hashes)
        print(f"Numbers of unique images considered during the process: {unique_images_count}")
        print(f"Overall number of images considered during the process: {total_images_count}")

    # Normal query to search t nearest neighbor
    def query(self, query_image_vector):
        return self.lsh.query(query_image_vector)

    # Extended query to search neighbor in case of size of result set is less than t
    def extended_query(self, query_image_vector):
        return self.lsh.extended_query(query_image_vector)

    # Function to print hash tables for debugging
    def describe(self):
        self.lsh.describe()
