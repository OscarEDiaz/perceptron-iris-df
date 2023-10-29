from sklearn.neural_network import MLPClassifier
from scipy.io.arff import loadarff
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import csv
import warnings

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
            # Each set should only contain data from that specific class
            shuffled_set = set.sample(frac = 1).reset_index(drop=True)

            shuffled_set_class = str(list(shuffled_set[shuffled_set.columns[-1]].drop_duplicates())[0])
            
            if shuffled_set_class == str(c):
                # Set tresholds
                BOTTOM = 0
                TOP = math.floor(len(shuffled_set) * self.training_percentage)

                # Retrieve data
                training_part = shuffled_set.iloc[BOTTOM:TOP]
                test_part = shuffled_set.iloc[TOP:len(shuffled_set)]
                
                # Append data to its corresponding part
                training_set = pd.concat([training_set, training_part])
                test_set = pd.concat([test_set, test_part])

            else:
                raise 'Set does not contain only one class'

        # Randomizing each set again to ensure that there is no bias
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

        clf = MLPClassifier(
            max_iter = epoch, 
            hidden_layer_sizes = topology,
            learning_rate_init = learning_rate, 
            activation = "logistic",
            momentum = momemtum,
            solver = 'lbfgs',
            random_state = 42
        )

        # Train the model
        x_train = training_set[train_attributes]
        y_train = training_set[train_class_attribute]

        x_test = test_set[test_attributes]
        y_test = test_set[test_class_attribute]

        warnings.filterwarnings("ignore")

        clf.fit(x_train, y_train)

        convergence = not clf.n_iter_ == clf.max_iter

        accuracy_train = 1 - clf.score(x_train, y_train) 
        accuracy_test = 1 - clf.score(x_test, y_test) 

        return accuracy_train, accuracy_test, convergence


    def calculate_mean_and_stdev(self, hyperparameters, minimal_fold_errors, k):
        for i, _ in enumerate (hyperparameters):
            minimal_fold_errors[i, -2:-1] = np.mean(minimal_fold_errors[i, 0:k])
            minimal_fold_errors[i, -1:] = np.std(minimal_fold_errors[i, 0:k])

    def get_best_experiment(self, hyperparameters, minimal_fold_errors):
        idx = np.argmin(minimal_fold_errors, axis=0)[-2]
        print('[-------------------------------------------------------------------]')
        print("The best experiment was achieved with the following hyperparameters:")
        print("# Epochs # Neurons Momemtum LearningRate")
        print(f'    {hyperparameters[idx]}')
        print("Mean Accuracy & Standard Desviation")
        print(f'    {minimal_fold_errors[idx, -2:]}')
        print('[-------------------------------------------------------------------]')


    def experimental_training(self, hyperparameters, performance_w_p, training_fold, test_fold, minimal_fold_errors, n):
        number_experiment = -1

        # For each experiment in the grid, retrieve the best model
        for exp in hyperparameters:
            number_experiment += 1
            train_errors, test_errors = [], []
            n_epochs = exp[0]

            # Performance window length
            p_window = int(math.floor(performance_w_p * n_epochs))

            overfitting_epoch = 0

            # Minimal error in the test set achieved
            min_test_error = np.inf

            #The last error where a significant improvment was seen
            last_significant_error = np.inf


            #Counter of the number of epochs without progress
            cont_no_progress = 0

            #The value of the minimal chande needed to consider a change as progress
            min_progress_need = 0.01

            # Train the model with n epochs
            for e in range(1, n_epochs+1):
                train_error, test_error, convergence = self.train_model(training_fold, test_fold, exp, e)

                # Determine the epoch where the test error was minimum.
                if test_error < min_test_error:
                    min_test_error = test_error
                    overfitting_epoch = e
                    
                train_errors.append(train_error)
                test_errors.append(test_error)
                
                change = abs(last_significant_error-test_error)

                # Verify for no progress during n epochs
                if change < min_progress_need:
                    cont_no_progress += 1
                else:
                    cont_no_progress = 0
                    last_significant_error = test_error

                # If there is no progress during p_window epochs, it get out of the loop
                if cont_no_progress > p_window:
                    break

            
            minimal_fold_errors[number_experiment, n] = min_test_error

            # plt.plot(range(1, len(train_errors) + 1), train_errors, label='Train Error')
            # plt.plot(range(1, len(test_errors) + 1), test_errors, label='Test Error')
            # plt.axvline(x=overfitting_epoch, color='red', linestyle='--', label='Overfitting epoch')
            # plt.xlabel('Número de Épocas')
            # plt.ylabel('Error')
            # plt.legend()
            # plt.show()

    def cross_validate(self, training_set, k, hyperparameters):
        percentage = 1/k
        fold_window = math.floor(len(training_set) * percentage)

        TOP = 0
        BOTTOM = 0

        # Creates a matrix to store the errors of each folds, the mean of these and the std dev
        minimal_fold_errors = np.zeros((len(hyperparameters), k+2))
        
        # Define window performance percentage to determine how many epochs will be analyzed
        performance_w_p = 0.1

        for n in range(0, k):
            # Define each fold length and limits using a sliding window
            TOP += fold_window
            BOTTOM = TOP - fold_window

            # Create the test fold
            test_fold = training_set.iloc[BOTTOM:TOP]

            # Define training fold as an empty dataframe
            training_fold = pd.DataFrame()
           
            # Assign empty training fold to the set complement
            if n != 0 and n < k:
                training_fold_complement_0 = training_set.iloc[0:BOTTOM]
                training_fold_complement_1 = training_set.iloc[TOP:len(training_set)]
                training_fold = pd.concat([training_fold_complement_0, training_fold_complement_1])
            elif n == 0:
                training_fold = training_set.iloc[TOP:len(training_set)]
            elif n == k:
                training_fold = training_set.iloc[0:BOTTOM]

            self.experimental_training(hyperparameters, performance_w_p, training_fold, test_fold, minimal_fold_errors, n)            

        self.calculate_mean_and_stdev(hyperparameters, minimal_fold_errors, k)
        self.get_best_experiment(hyperparameters, minimal_fold_errors)


    def read_csv_hyperparameters(self, filename):
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
    hyperparameters = model.read_csv_hyperparameters('test.csv')

    performance_w_p = 0.1
    minimal_fold_errors = np.zeros((len(hyperparameters), 3))

    # Cross validate the model
    print('[-------------------------CROSS VALIDATION-------------------------]')
    model.cross_validate(training_set, 3, hyperparameters)
    
    print('[------------------------MODEL TRAINING------------------------]')
    model.experimental_training(hyperparameters, performance_w_p, training_set, test_set, minimal_fold_errors, 0)
    model.calculate_mean_and_stdev(hyperparameters, minimal_fold_errors, 3)
    model.get_best_experiment(hyperparameters, minimal_fold_errors)

main_pipeline()