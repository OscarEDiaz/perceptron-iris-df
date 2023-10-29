from sklearn import datasets
from sklearn.neural_network import MLPClassifier
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

        print({'topology': topology, 'momemtum': momemtum, 'learning rate': learning_rate, 'epoch': epoch})

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

        clf.fit(x_train, y_train)

        convergence = not clf.n_iter_ == clf.max_iter

        accuracy_train = 1 - clf.score(x_train, y_train) 
        accuracy_test = 1 - clf.score(x_test, y_test) 

        return accuracy_train, accuracy_test, convergence


    def cross_validate(self, training_set, k, hyperparameters):
        percentage = 1/k
        fold_window = math.floor(len(training_set) * percentage)

        TOP = 0
        BOTTOM = 0

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

            # print(f'--------------FOLD: {n}-----------------')
            # print('Training fold: ')
            # print(training_fold)
            # print('Testing fold: ')
            # print(test_fold)
            # print('-----------------------------------------')

            # Define window performance percentage to determine how many epochs will be analyzed
            performance_w_p = 0.05

            # Standard deviation treshold (x times the mean)
            std_dev_trsh = 1.5

            # For each experiment in the grid, retrieve the best model
            for exp in hyperparameters:
                train_errors, test_errors = [], []
                n_epochs = exp[0]

                # Performance window length
                p_window = int(math.floor(performance_w_p * n_epochs))

                # Define training and test windows
                tr_top, tr_bottom = 0, 0

                tst_top, tst_bottom = 0, 0
                overfitting_epoch = 0

                flag = 0
                enable = True

                # Train the model with n epochs
                for e in range(1, n_epochs+1):
                    train_error, test_error, convergence = self.train_model(training_fold, test_fold, exp, e)

                    train_errors.append(train_error)
                    test_errors.append(test_error)


                    # Train erros and test erros have enough items to fill the window
                    if len(train_errors) >= p_window and len(test_errors) >= p_window:
                        tr_top = p_window
                        tr_bottom = tr_top - p_window

                        tst_top = p_window
                        tst_bottom = tst_top - p_window

                        print({'tr_top': tr_top, 'tr_bottom': tr_bottom, 'tst_top': tst_top, 'tst_bottom': tst_bottom})

                        # Define the moving averages of both windows to visualize their trend
                        train_window = pd.Series(train_errors[tr_bottom:tr_top])
                        train_moving_avg = train_window.rolling(p_window).mean()

                        test_window = pd.Series(test_errors[tst_bottom:tst_top])
                        test_moving_avg = test_window.rolling(p_window).mean()

                        # Determine if there's any improvement on both sets
                        if convergence and enable:
                            print('It converged')
                            enable = False
                            train_is_descending = train_moving_avg.iloc[-1] < train_moving_avg.iloc[0]
                            test_is_ascending = test_moving_avg.iloc[-1] > test_moving_avg.iloc[0]

                            # Calculate the standard deviation to see if there's an erratic behaviour or not
                            train_std_dev = np.std(list(train_window))
                            test_std_dev = np.std(list(test_window))

                            tr_std_d_treshold = std_dev_trsh * np.mean(list(train_window))
                            tst_std_d_treshold = std_dev_trsh * np.mean(list(test_window))

                            # Define the conditions
                            tr_is_erratic = train_std_dev > tr_std_d_treshold
                            tst_is_erratic = test_std_dev > tst_std_d_treshold

                            train_is_improving = train_is_descending and not tr_is_erratic
                            test_is_improving = test_is_ascending and not tst_is_erratic

                            if not train_is_improving or not test_is_improving:
                                flag += 1
                                overfitting_epoch = e
                        else:
                            print('Did not converged')

                # print(len(train_errors))
                # print(len(test_errors))
                print('flag: ', flag)
                print('overfitting epoch: ', overfitting_epoch)

                plt.plot(range(1, len(train_errors) + 1), train_errors, label='Train Error')
                plt.plot(range(1, len(test_errors) + 1), test_errors, label='Test Error')
                plt.axvline(x=overfitting_epoch, color='red', linestyle='--', label='Overfitting epoch')
                plt.xlabel('Número de Épocas')
                plt.ylabel('Error')
                plt.legend()
                plt.show()

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

    # print(hyperparameters)

    # Cross validate the model
    model.cross_validate(training_set, 3, hyperparameters)


main_pipeline()