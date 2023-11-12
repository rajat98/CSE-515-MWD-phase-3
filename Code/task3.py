import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import re
from tqdm import tqdm


# Your parse_string function
def parse_string(string):
    values = re.findall(r'-?\d+\.\d+', string)
    np_array = np.array(values, dtype=float)
    return torch.tensor(np_array, dtype=torch.float32)


class KNeighborsClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in tqdm(X)]
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.n_neighbors]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common


class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value


class DecisionTreeClassifiers():
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features, curr_depth)
            if best_split["info_gain"] > 0:
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["info_gain"])

        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features, curr_depth):
        best_split = {}
        max_info_gain = -float("inf")

        for feature_index in tqdm(range(num_features), desc=f"Depth {curr_depth}"):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)

            for threshold in possible_thresholds[:5]:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)

                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")

                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        return best_split

    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child, mode="gini"):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode == "gini":
            gain = self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain

    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls ** 2
        return 1 - gini

    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)

    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_" + str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        dataset = np.concatenate((X, Y.reshape(-1, 1)), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        from sklearn.tree import DecisionTreeClassifier
        dt_class = DecisionTreeClassifier()
        dt_class.fit(even_features, even_labels)
        predictions = dt_class.predict(odd_features)
        return predictions

    def make_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)


class PPRClassifier():
    def __init__(self, n_neighbors=3, randomization_factor=0.1):
        self.n_neighbors = n_neighbors
        self.randomization_factor = randomization_factor
        self.knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

    def preprocess_data(self, X):
        X = np.array(X)
        rows, cols = X.shape
        randomization_matrix = np.random.normal(loc=1.0, scale=self.randomization_factor, size=(rows, cols))
        return X * randomization_matrix

    def fit(self, X, y):
        X = np.array(X)
        preprocessed_X = self.preprocess_data(X)
        self.knn_classifier.fit(preprocessed_X, y)

    def predict(self, X):
        preprocessed_X = self.preprocess_data(X)
        return self.knn_classifier.predict(preprocessed_X)


# Function to print metrics for each classifier
def calculate_metrics(predictions, labels):
    unique_labels = set(labels)
    metrics_per_label = {}

    for label in unique_labels:
        true_positive = sum((p == label) and (p == l) for p, l in zip(predictions, labels))
        false_positive = sum((p == label) and (p != l) for p, l in zip(predictions, labels))
        false_negative = sum((p != label) and (p != l) for p, l in zip(predictions, labels))

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        metrics_per_label[label] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1
        }

    accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(labels) if len(labels) > 0 else 0

    return metrics_per_label, accuracy


def print_metrics(classifier_name, predictions, labels):
    metrics, accuracy = calculate_metrics(predictions, labels)

    print(f"{classifier_name} Metrics:")
    for label, metrics_dict in metrics.items():
        print(f"Label {label} - Precision: {metrics_dict['precision']:.2f}, Recall: {metrics_dict['recall']:.2f}, F1-score: {metrics_dict['f1-score']:.2f}")

    print(f"Overall Accuracy: {accuracy:.2f}\n")



if __name__ == '__main__':
    option = int(input('Please input classifier\n1.m-NN Classifier\n2.Decision Tree Classifier\n3.PPR Classifier\n'))
    op = input('Do you want to find out prediction for any specific odd image?[y/n]')
    image_id = None
    if op == 'y':
        image_id = int(input('Please enter image id: '))
        if image_id % 2 != 1:
            print('Invalid image id presented.')
            exit(1)

    # Read the features from the CSV file using pandas
    df = pd.read_csv('./../FD_Objects_all.csv')

    # Apply the parse_string function to the desired column
    df['ResNet_FC_1000'] = df['ResNet_FC_1000'].apply(parse_string)

    # Split the data into even and odd subsets
    even_data = df.iloc[::2]
    odd_data = df.iloc[1::2]

    # Extract features and labels
    even_features = list(even_data['ResNet_FC_1000'])
    even_labels = list(even_data['Labels'])
    odd_features = list(odd_data['ResNet_FC_1000'])
    odd_labels = list(odd_data['Labels'])

    if option == 1:
        m_value = int(input('Please choose value of m: '))
        m_nn_classifier = KNeighborsClassifier(n_neighbors=m_value)
        m_nn_classifier.fit(even_features, even_labels)
        m_nn_predictions = m_nn_classifier.predict(odd_features)
        if op == 'y':
            print(f'\nPrediction for provided image is:{m_nn_predictions[image_id//2]}\n')
        print_metrics("m-NN Classifier", m_nn_predictions, odd_labels)
    elif option == 2:
        dt_classifier = DecisionTreeClassifiers(min_samples_split=3, max_depth=3)
        dt_classifier.fit(even_features, even_labels)
        dt_predictions = dt_classifier.predict(odd_features)
        if op == 'y':
            print(f'\nPrediction for provided image is:{dt_predictions[image_id//2]}')
        print_metrics("Decision Tree Classifier", dt_predictions, odd_labels)
    elif option == 3:
        ppr_classifier = PPRClassifier(n_neighbors=5, randomization_factor=0.5)
        ppr_classifier.fit(even_features, even_labels)
        ppr_predictions = ppr_classifier.predict(odd_features)
        if op == 'y':
            print(f'\nPrediction for provided image is:{ppr_predictions[image_id // 2]}')
        print_metrics("PPR Classifier", ppr_predictions, odd_labels)
    else:
        print('Please choose valid option')
