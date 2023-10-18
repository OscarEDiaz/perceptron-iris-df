from sklearn import datasets
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

class Model:
    def __init__(self, dataset, training_percentage):
        self.training_percentage = training_percentage
        self.testing_percentage = 1 - training_percentage
        self.dataset = dataset


    # Creation of training and test sets 
    # Uses random stratified sampling
    def define_training_test(self):
        # Separate by classes
        class_column = self.dataset.columns[-1]
        classes = self.dataset[self.database.columns[-1]].drop_duplicates()

        print('Lenght of classes', len(classes))

        training_set = pd.DataFrame()
        test_set = pd.DataFrame()

        for c in classes:
            mask = self.dataset[class_column] == c
            set = self.dataset[mask]
            
            # Randomize each set
            shuffled_set = set.sample(frac = 1)

            # Set tresholds
            BOTTOM = 0
            TOP = math.floor(len(shuffled_set) * self.training_percentage)

            # Retrieve data
            training_part = shuffled_set.iloc[BOTTOM:TOP]
            test_part = shuffled_set.iloc[TOP:len(shuffled_set)]
            
            # Append data to its corresponding part
            training_set = pd.concat([training_set, training_part])
            test_set = pd.concat([test_set, test_part])

        return training_set, test_set


    def train(self, training_set, test_set, epochs, n_layers, n_neurons, l_rate, m):
        nn_arquitecture = [n_neurons] * n_layers

        clf = MLPClassifier(max_iter = epochs, 
                            hidden_layer_sizes = (nn_arquitecture),
                            learning_rate = l_rate, 
                            momentum = m)

        # Train the model
        clf.fit(training_set[:-1], training_set[-1])

        accuracy = clf.score(test_set[:-1], test_set[-1])

        return accuracy



    def cross_validate(self, training_set, n_folds, hyperparameters):
        # Hyperparameters
        epochs        = hyperparameters[0]
        hidden_layers = hyperparameters[1]
        neurons       = hyperparameters[2]
        learning_rate = hyperparameters[3]
        momemtum      = hyperparameters[4]

        range = 1/n_folds

        for n in n_folds:
            # Sliding window
            TOP += range
            BOTTOM = TOP - range

            # Create test fold
            test_fold = training_set.iloc[BOTTOM:TOP]
            training_fold = pd.DataFrame()

            # Create training fold
            if n != 0 and n < len(n_folds) - 1:
                training_fold_complement_0 = training_set.iloc[0:BOTTOM]
                training_fold_complement_1 = training_set.iloc[TOP:len(training_set)]
                training_fold = pd.concat([training_fold_complement_0, training_fold_complement_1])
            elif n == 0:
                training_fold = training_set.iloc[TOP:len(training_set)]
            elif n == len(n_folds) - 1:
                training_fold = training_set.iloc[0:BOTTOM]

            # Train the model and retrieve its accuracy
            accuracy = self.train(training_fold, test_fold, epochs, hidden_layers, neurons, learning_rate, momemtum)

            # Print the accuracy
            self.print_results(accuracy=accuracy)


    def print_results(self, accuracy):
        print(f'Train Results')
        print(f'---------------------')
        print(f'Accuracy  = {accuracy}')
        print(f'---------------------')

# Load iris database
iris = datasets.load_iris()

# Convert the iris dataset to a pandas dataframe
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the target variable to the dataframe
df['target'] = iris.target

# Create the model
model = Model(df, 0.7)

# Create training and test sets
training_set, test_set = model.define_training_test()

# Define the hyperparameters
# EPOCHS / HIDDEN LAYERS / NEURONS / LEARNING RATE / MOMEMTUM
hpms = [5, 3, 4, 0.3, 0.2]

# Cross validate the model
model.cross_validate(3, hpms)
