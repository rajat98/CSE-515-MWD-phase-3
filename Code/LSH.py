import os
import pickle
from numpy.random import randn
from torchvision import datasets

from utilities import hash_func, generate_one_bit_diff_numbers, get_dataset_feature_set, get_input_image_vector, \
    BASE_DIR, DATABASE, calculate_euclidian_distance, plot_result, get_unique_images_count


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


def get_pruned_result_set(result_set, l, t):
    image_id_to_result_count_dict = {}
    for result in result_set:
        image_id = result["label"]
        if image_id not in image_id_to_result_count_dict.keys():
            image_id_to_result_count_dict[image_id] = 0
        image_id_to_result_count_dict[image_id] += 1
    sorted_image_id_frequency_list = sorted(image_id_to_result_count_dict.items(), key=lambda item: item[1],
                                            reverse=True)
    result_set = []
    confidence = 0
    if len(sorted_image_id_frequency_list) > 1:
        i = 0
        freq = 0
        for i in range(t):
            result_set.append({"label": sorted_image_id_frequency_list[i][0]})
            freq = sorted_image_id_frequency_list[i][1]
            confidence += freq
        i += 1
        while i < len(sorted_image_id_frequency_list) and sorted_image_id_frequency_list[i][1] == freq:
            result_set.append({"label": sorted_image_id_frequency_list[i][0]})
            i += 1
            confidence += freq
    confidence /= l
    confidence /= t
    return result_set, confidence


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

    def find_t_nearest_neighbor(self, input_image_id_or_path, t, approach=2):
        query_image_vector = get_input_image_vector(input_image_id_or_path)
        dataset = datasets.Caltech101(BASE_DIR)
        collection = DATABASE.feature_descriptors
        result_set = self.query(query_image_vector)
        result_set = [item for sublist in result_set for item in sublist]

        # approach 1: calculate euclidian of t that have the highest occurrences in all tables(less # of comparisons)
        if approach == 1:
            result_set, confidence = get_pruned_result_set(result_set, self.layers, t)
            print(f"Original Approach Confidence: {confidence}")
            # neighbor bucket
            if confidence < 1:
                extended_result_set = self.extended_query(query_image_vector)
                result_set = [item for sublist in extended_result_set for item in sublist]
                result_set, confidence = get_pruned_result_set(result_set, self.layers, t)
                print(f"Extended Approach Confidence: {confidence}")

        # approach 2: calculate euclidian of l*h(compare result set as is)
        euclidian_distance_list = []
        for result in result_set:
            # for entry in table_result_set:
            image_id = result["label"]
            image_vector = collection.find_one({"image_id": image_id})["resnet_fc_1000"]
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

    def query(self, query_image_vector):
        return self.lsh.query(query_image_vector)

    def extended_query(self, query_image_vector):
        return self.lsh.extended_query(query_image_vector)

    def describe(self):
        self.lsh.describe()
