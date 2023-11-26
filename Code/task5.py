from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def retrieve_results():
    file_path = '../Outputs/T4/task4.csv'
    
    data = pd.read_csv(file_path)
    return data

def get_user_tags(initial_results):
    initial_results['User Tag'] = ""

    while True:
        print(initial_results)

        imageid = input("Enter image id for user tag or 'q' to quit: ")

        if imageid == 'q':
            return initial_results

        print("Select an option:")
        print("1 - Very Relevant (R+)")
        print("2 - Relevant (R)")
        print("3 - Irrelevant (I)")
        print("4 - Very Irrelevant (I-)")
        print("5 - Stop")

        choice = input("Enter your choice of user tag: ")

        if choice == '1':
            print(f"You selected Very Relevant (R+) for Image ID: {imageid}")
            initial_results.loc[initial_results['Image ID'] == int(imageid), 'User Tag'] = 'R+'
        elif choice == '2':
            print(f"You selected Relevant (R) for Image ID: {imageid}")
            initial_results.loc[initial_results['Image ID'] == int(imageid), 'User Tag'] = 'R'
        elif choice == '3':
            print(f"You selected Irrelevant (I) for Image ID: {imageid}")
            initial_results.loc[initial_results['Image ID'] == int(imageid), 'User Tag'] = 'I'
        elif choice == '4':
            print(f"You selected Very Irrelevant (I-) for Image ID: {imageid}")
            initial_results.loc[initial_results['Image ID'] == int(imageid), 'User Tag'] = 'I-'
        elif choice == '5':
            return initial_results
        else:
            print("Invalid choice. Please enter a valid option.")


def train_svm(tagged_results):
    tagged_results['User Tag'] = tagged_results['User Tag'].replace("", 'N')
    
    # Extract features and labels from tagged_results
    features = tagged_results['Euclidean Distance'].values.reshape(-1, 1)
    labels = tagged_results['User Tag'].values

    # Map labels to numerical values
    label_mapping = {"R+": 2, "R": 1, "N": 0, "I": -1, "I-": -2}
    labels = np.array([label_mapping.get(label, 0) for label in labels])

    # Initialize SVM model
    svm_model = svm.SVC()

    # Train SVM model
    svm_model.fit(features, labels)

    return svm_model

def rank_svm_results(svm_model, initial_results):
    # Extract features from initial_results
    features = initial_results['Euclidean Distance'].values.reshape(-1, 1)

    # Use the SVM model to predict the relevance of each result
    predicted_relevance = svm_model.predict(features)

    # Add the predicted relevance to the initial_results DataFrame
    initial_results['Predicted Relevance'] = predicted_relevance

    user_int = initial_results
    integer_mapping = {'R+': 2, 'R': 1, 'N': 0, 'I': -1, 'I-': -2}
    user_int['User Tag Integer'] = user_int['User Tag'].map(integer_mapping)

    weight_predicted_relevance = 0.3
    weight_user_tag = 0.3
    weight_distance = 0.4

    # Calculate a score for each result based on the weighted sum of the predicted relevance, the user tag, and the Euclidean distance
    initial_results['Score'] = weight_predicted_relevance * user_int['Predicted Relevance'] + weight_user_tag * user_int['User Tag Integer'] - weight_distance * user_int['Euclidean Distance']

    min_score = initial_results['Score'].min()
    initial_results['Score'] += abs(min_score) + 1

    # Sort the results by the score
    ranked_results = initial_results.sort_values(by='Score', ascending=False)

    filtered_columns = ['Image ID', 'Euclidean Distance', 'User Tag', 'Score']

    return ranked_results[filtered_columns]



def rank_probabilistic_results(relevance_probabilities, label_mapping, initial_results):
    # Convert relevance probabilities to a DataFrame
    # The columns are determined by the keys of label_mapping
    relevance_df = pd.DataFrame(relevance_probabilities, columns=label_mapping.keys())

    # Calculate a relevance score for each result
    # This could be a simple sum of the probabilities for 'R' and 'R+' tags if they exist
    relevant_columns = [col for col in ['R', 'R+'] if col in relevance_df.columns]
    relevance_score = relevance_df[relevant_columns].sum(axis=1)

    # Define weights for the relevance score, the user tag, and the Euclidean distance
    # These weights can be adjusted based on how much importance you want to give to each factor
    weight_relevance_score = 0.4
    weight_user_tag = 0.3
    weight_distance = 0.3

    # Map user tags to numerical values
    user_int = initial_results
    integer_mapping = {'R+': 2, 'R': 1, 'N': 0, 'I': -1, 'I-': -2}
    user_int['User Tag Integer'] = user_int['User Tag'].map(integer_mapping)

    # Calculate a score for each result based on the weighted sum of the relevance score, the user tag, and the Euclidean distance
    initial_results['Score'] = weight_relevance_score * relevance_score + weight_user_tag * user_int['User Tag Integer'] - weight_distance * initial_results['Euclidean Distance']

    # Shift all scores by the absolute value of the minimum score plus 1
    min_score = initial_results['Score'].min()
    initial_results['Score'] += abs(min_score) + 1

    # Sort the results by the score
    ranked_results = initial_results.sort_values(by='Score', ascending=False)

    filtered_columns = ['Image ID', 'Euclidean Distance', 'User Tag', 'Score']

    return ranked_results[filtered_columns]



def estimate_relevance_probabilities(tagged_results):
    tagged_results['User Tag'] = tagged_results['User Tag'].replace("", 'N')
    
    # Extract features and labels from tagged_results
    features = tagged_results['Euclidean Distance'].values.reshape(-1, 1)
    labels = tagged_results['User Tag'].values

    # Get unique labels and map them to numerical values
    unique_labels = np.unique(labels)
    label_mapping = {label: i for i, label in enumerate(unique_labels)}
    labels = np.array([label_mapping[label] for label in labels])

    # Initialize logistic regression model
    lr_model = LogisticRegression()

    # Train logistic regression model
    lr_model.fit(features, labels)

    # Estimate probabilities of each class
    relevance_probabilities = lr_model.predict_proba(features)

    return relevance_probabilities, label_mapping


def svm_based_feedback_system():
    #Retrieve initial results
    initial_results = retrieve_results()
    
    #User tags some results
    tagged_results = get_user_tags(initial_results)
    
    #Learn from tagged samples and update training set
    svm_model = train_svm(tagged_results)
        
    #Return new set of ranked results
    new_results = rank_svm_results(svm_model, initial_results)
    
    return new_results


def probabilistic_feedback_system():
    #Retrieve initial results
    initial_results = retrieve_results()
    
    #User tags some results
    tagged_results = get_user_tags(initial_results)
    
    #Estimate probability of relevance for each document
    relevance_probabilities, label_mapping = estimate_relevance_probabilities(tagged_results)
    
    #Return new set of ranked results
    new_results = rank_probabilistic_results(relevance_probabilities, label_mapping, initial_results)
    
    return new_results


if __name__ == "__main__":
    print("Select an option:")
    print("1 - SVM based relevance feedback system")
    print("2 - probabilistic relevance feedback system")

    choice = input("Enter your choice: ")

    if choice == '1':
        results = svm_based_feedback_system()
    elif choice == '2':
        results = probabilistic_feedback_system()
    else:
        print("Invalid choice. Quitting")

    print(results)