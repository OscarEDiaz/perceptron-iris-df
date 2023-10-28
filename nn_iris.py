from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from scipy.io.arff import loadarff
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


    def train_model(self, training_set, test_set, hyperparameters, epoch):
        # Training set attributes definition
        train_attributes = training_set.columns[:-1]
        train_class_attribute = training_set.columns[-1]

        # Test set attributes definition
        test_attributes = test_set.columns[:-1]
        test_class_attribute = test_set.columns[-1]

        # Define the hyperparameters
        topology      = hyperparameters[1]
        momemtum      = hyperparameters[2]
        learning_rate = hyperparameters[3]

        # print({'topology': topology, 'momemtum': momemtum, 'learning rate': learning_rate, 'epoch': epoch})

        clf = MLPClassifier(max_iter = epoch, 
                            hidden_layer_sizes = topology,
                            learning_rate_init = learning_rate, 
                            activation="logistic",
                            momentum = momemtum,
                            solver='lbfgs',
                            random_state=42)

        # Train the model
        x_train = training_set[train_attributes]
        y_train = training_set[train_class_attribute]

        x_test = test_set[test_attributes]
        y_test = test_set[test_class_attribute]

        clf.fit(x_train, y_train)

        accuracy_train = clf.score(x_train, y_train) 
        accuracy_test = clf.score(x_test, y_test) 

        return [accuracy_train, accuracy_test]


    def cross_validate(self, training_set, k, hyperparameters):
        percentage = 1/k

        TOP = 0
        BOTTOM = 0

        for n in range(0, k):
            window = math.floor(len(training_set) * percentage)

            # Sliding window
            TOP += window
            BOTTOM = TOP - window

            # Create test fold
            test_fold = training_set.iloc[BOTTOM:TOP]
            training_fold = pd.DataFrame()

            # Create training fold
            if n != 0 and n < k:
                training_fold_complement_0 = training_set.iloc[0:BOTTOM]
                training_fold_complement_1 = training_set.iloc[TOP:len(training_set)]
                training_fold = pd.concat([training_fold_complement_0, training_fold_complement_1])
            elif n == 0:
                training_fold = training_set.iloc[TOP:len(training_set)]
            elif n == k:
                training_fold = training_set.iloc[0:BOTTOM]

            print(f'--------------FOLD: {n}-----------------')
            print('Training fold: ')
            print(training_fold)
            print('Testing fold: ')
            print(test_fold)
            print('-----------------------------------------')

            # For each experiment in the grid, retrieve the best model
            for exp in hyperparameters:
                train_errors, test_errors = [], []
                n_epochs = exp[0]

                for e in range(1, n_epochs+1):
                    errors = self.train_cross_validation(training_fold, test_fold, exp, e)

                    train_errors.append(errors[0])
                    test_errors.append(errors[1])

                # print(len(train_errors))
                # print(len(test_errors))

                plt.plot(range(1, len(train_errors) + 1), train_errors, label='Train Error')
                plt.plot(range(1, len(test_errors) + 1), test_errors, label='Test Error')
                plt.xlabel('Número de Épocas')
                plt.ylabel('Error')
                plt.legend()
                plt.show()

    def read_csv(self, filename):
        def list_to_int(x):
            if type(x) is list:
                return [int(n) for n in x]
            else:
                return float(x)

        try:
            with open(filename, newline='') as csvfile:
                reader = list(csv.reader(csvfile, delimiter=','))
                
                # Remove the first row (column headers)
                reader = reader[1:]

                # CSV preprocessing
                for i, row in enumerate(reader):
                    # Retrieve the neurons array for each epoch
                    str = row[1]
                    neurons_array = str.split(';')

                    row[1] = neurons_array

                    row = list(map(list_to_int, row))

                    # Cast to int the number of epochs
                    row[0] = int(row[0])

                    reader[i] = row

            return reader
        except EnvironmentError:
            raise


    def print_results(self, accuracy):
        print(f'Train Results')
        print(f'---------------------')
        print(f'Accuracy  = {accuracy}')
        print(f'---------------------')


def main_pipeline():
    # Load iris augmented database
    raw_data = loadarff('irisAumentedData.arff')
    df_data = pd.DataFrame(raw_data[0])

    # Change the nominal values of variety to numeric
    df_data['variety'] = pd.factorize(df_data['variety'])[0]


    # Create the model
    model = Model(df_data, 0.7)

    # Create training and test sets
    training_set, test_set = model.define_training_test()

    # Retrieve the hyperparameters grid from a csv file
    hyperparameters = model.read_csv('test.csv')

    # print(hyperparameters)

    # Cross validate the model
    model.cross_validate(training_set, 3, hyperparameters)


main_pipeline()
