from sklearn import datasets
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import csv

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
        classes = self.dataset[self.dataset.columns[-1]].drop_duplicates()

        training_set = pd.DataFrame()
        test_set = pd.DataFrame()

        for c in classes:
            mask = self.dataset[class_column] == c
            set = self.dataset[mask]
            
            # Randomize each set
            shuffled_set = set.sample(frac = 1).reset_index(drop=True)

            # Set tresholds
            BOTTOM = 0
            TOP = math.floor(len(shuffled_set) * self.training_percentage)

            # Retrieve data
            training_part = shuffled_set.iloc[BOTTOM:TOP]
            test_part = shuffled_set.iloc[TOP:len(shuffled_set)]
            
            # Append data to its corresponding part
            training_set = pd.concat([training_set, training_part])
            test_set = pd.concat([test_set, test_part])

        training_set = training_set.sample(frac = 1)
        test_set = test_set.sample(frac = 1)
        return training_set, test_set


    def train(self, training_set, test_set, epochs, topology, learning_rate, momemtum):
        # Training set attributes definition
        train_attributes = training_set.columns[:-1]
        train_class_attribute = training_set.columns[-1]

        # Test set attributes definition
        test_attributes = test_set.columns[:-1]
        test_class_attribute = test_set.columns[-1]


        clf = MLPClassifier(max_iter = epochs, 
                            hidden_layer_sizes = topology,
                            learning_rate_init = learning_rate, 
                            momentum = momemtum)

        # Train the model
        clf.fit(training_set[train_attributes], training_set[train_class_attribute])

        accuracy = clf.score(test_set[test_attributes], test_set[test_class_attribute])

        return accuracy


    def cross_validate(self, training_set, n_folds, hyperparameters):
        # Hyperparameters
        epochs        = hyperparameters[0]
        topology      = hyperparameters[1]
        learning_rate = hyperparameters[2]
        momemtum      = hyperparameters[3]

        percentage = 1/n_folds

        TOP = 0
        BOTTOM = 0

        for n in range(0, n_folds):
            window = math.floor(len(training_set) * percentage)

            # Sliding window
            TOP += window
            BOTTOM = TOP - window

            # Create test fold
            test_fold = training_set.iloc[BOTTOM:TOP]
            training_fold = pd.DataFrame()

            # Create training fold
            if n != 0 and n < n_folds:
                training_fold_complement_0 = training_set.iloc[0:BOTTOM]
                training_fold_complement_1 = training_set.iloc[TOP:len(training_set)]
                training_fold = pd.concat([training_fold_complement_0, training_fold_complement_1])
            elif n == 0:
                training_fold = training_set.iloc[TOP:len(training_set)]
            elif n == n_folds:
                training_fold = training_set.iloc[0:BOTTOM]

            # Train the model and retrieve its accuracy
            accuracy = self.train(training_fold, test_fold, epochs, topology, learning_rate, momemtum)

            # Print the accuracy
            self.print_results(accuracy=accuracy)


    def read_csv(self, filename):
        def list_to_int(x):
            if type(x) is list:
                return [float(n) for n in x]
            else:
                return float(x)

        try:
            with open(filename, newline='') as csvfile:
                reader = list(csv.reader(csvfile, delimiter=','))
                
                # Remove the first row
                reader = reader[1:]

                # Process the number of neurons for each epoch
                for row in reader:
                    str = row[3]
                    neurons_array = str.split(';')

                    row[3] = neurons_array

                    row = list(map(list_to_int, row))
            
            return reader
        except EnvironmentError:
            raise


    def print_results(self, accuracy):
        print(f'Train Results')
        print(f'---------------------')
        print(f'Accuracy  = {accuracy}')
        print(f'---------------------')


def main_pipeline():
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
    hyperparameters = model.read_csv('tet.csv')

    # Cross validate the model
    # model.cross_validate(training_set=training_set, n_folds=3, hyperparameters=hpms)


main_pipeline()