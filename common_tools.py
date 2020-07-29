# methods used in multiple other files. Most are shared between voltage_model and current_model


import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV


# creates a scaler with the help of the train subset, uses the scaler on the current test subset.
# type_of_scaler determines if it was called to save a scaler and if so, what kind of scaler
def standardize_train_test_data(x_train, x_test, type_of_scaler):
    scaler = StandardScaler()
    standard_x_train = scaler.fit_transform(x_train)
    standard_x_test = scaler.transform(x_test)

    if type_of_scaler == 1:
        pickle.dump(scaler, open("Models/current_scaler.pkl", "wb"))
    elif type_of_scaler == 0:
        pickle.dump(scaler, open("Models/voltage_scaler.pkl", "wb"))
    else:
        return standard_x_train, standard_x_test


# pca analysis of the input
def get_pca(x_in):
    x_standardized = StandardScaler().fit_transform(x_in)
    pca = PCA()
    pca.fit(x_standardized)
    print("PCA variance ratios for all:", pca.explained_variance_ratio_)
    print(pd.DataFrame(pca.components_, columns=x_in.columns))


# returns the mean squared error of the test data subset. For a current model, rounds to 2 decimals
def train_test_mean_error(model_kind, model, x_train, x_test, y_train, y_test):
    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_error = mean_squared_error(y_train, train_pred)
    test_error = mean_squared_error(y_test, test_pred)

    if model_kind == "current":
        train_error = train_error.round(2)
        test_error = test_error.round(2)

    print("train error", train_error, "test error", test_error)
    return test_error


# plots the actual values and the predicted values on the same graph
def real_and_predicted_plots(x_test_values, y_actual, y_prediction, ylabel, mae):
    time_axis = [0]

    # different parameters are required for each kind of graph, voltage or current
    if ylabel == "Voltage (V)":
        upper_limit = 201
        y_actual_with_initial = pd.Series([1.1]).append(y_actual)
        y_prediction_with_initial = pd.Series([1.1]).append(y_prediction)
        mae_text = "Mean Absolute Error: " + str(mae.round(8)) + "V"
        plt.text(50, 0.96, mae_text)
        plt.axis([0, 200, 0.95, 1.12])
        actual_plot_style = "b-"
        redicted_plot_style = "r-"
    else:
        upper_limit = 41
        y_actual_with_initial = pd.Series([0]).append(y_actual)
        y_prediction_with_initial = pd.Series([0]).append(y_prediction)
        mae_text = "Mean Absolute Error: " + str(mae.round(2)) + "uA"
        plt.text(5, -40, mae_text)
        plt.axis([0, 20, -50, 75])
        actual_plot_style = "b-D"   # the plots of the currents indicate the actual values
        redicted_plot_style = "r-o"

    for k in range(1, upper_limit):
        if upper_limit == 41:
            time_axis.append(k/2)
        else:
            time_axis.append(k)

    actual_line, = plt.plot(time_axis, y_actual_with_initial, actual_plot_style, markersize=5)
    predicted_line, = plt.plot(time_axis, y_prediction_with_initial, redicted_plot_style, markersize=4)
    title = x_test_values
    plt.legend((actual_line, predicted_line), ("actual", "predicted"))
    plt.title(title)
    plt.xlabel("Time (ps)")
    plt.ylabel(ylabel)
    plt.show()


# compares the actual simulation values and the predicted ones. Prints the mean absolute error of each
# and the average mean absolute error across the entire test subset. Can also be used to show the plots
def compare_real_and_predicted(x_test, y_test, test_prediction, plot_ylabel):
    average_mae = 0
    less_than_two = 0
    two_to_five = 0
    more_than_five = 0
    for i in range(0, y_test.shape[0]):  # for each row of data in the test dataset

        y_test_row = y_test.iloc[i]
        prediction_row = pd.Series(test_prediction[i])

        mae = mean_absolute_error(y_test_row, prediction_row)
        average_mae += mae

        if mae < 2:
            less_than_two += 1
        elif 2 < mae < 5:
            two_to_five += 1
        else:
            more_than_five += 1

        ## these three cases can be used to create the respective kinds of plots
        ## note: all voltage cases have mae < 2
        # if mae < 2:
        #     print("less than two")
        #     real_and_predicted_plots(format_x_test_string_data(x_test.iloc[i]), y_test_row,
        #                              prediction_row, plot_ylabel,  mae)
        # # if 2 < mae < 5:
        #     print("two to five")
        #     real_and_predicted_plots(format_x_test_string_data(x_test.iloc[i]), y_test_row,
        #                              prediction_row, plot_ylabel, mae)
        # if mae > 5:
        #     print("more than five")
        #     real_and_predicted_plots(format_x_test_string_data(x_test.iloc[i]), y_test_row,
        #                              prediction_row, plot_ylabel, mae)


    print("MAE of less than 2:", round(((less_than_two*100)/y_test.shape[0]), 2), "%\nbetween 2 and 5:",
          round(((two_to_five*100)/y_test.shape[0]), 2), "%\nmore than 5:",
          round(((more_than_five*100)/y_test.shape[0]), 2), "%")
    average_mae = average_mae / y_test.shape[0]
    print("average mean absolute error:", average_mae)


# grid search for the hyperparameters of the model. Repeated fitting of
# the standardized input across the grid of possible hyperparameter values provided
def grid_search(estimator, grid, x, y):
    sc = StandardScaler()
    x = sc.fit_transform(x)

    clf = GridSearchCV(estimator, param_grid=grid, cv=5)
    clf.fit(x, y)
    print(clf.best_params_)
    number_of_estimators = sum(clf.best_params_.values())

    return number_of_estimators


# saves the model using pickle
def save_model(model, x_test, y_test, model_path, x_test_path, y_test_path):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    x_test_txt = open(x_test_path, "w")
    y_test_txt = open(y_test_path, "w")
    x_test_txt.write(x_test.to_string())
    y_test_txt.write(y_test.to_string())
    x_test_txt.close()
    y_test_txt.close()


# receives x_test.iloc[i] as parameter, the input data for a single simulation. Returns the input values as a string
def format_x_test_string_data(x_test):
    formatted_string = "C: " + str(x_test.iat[0]) + " T1: " + str(x_test.iat[1]) + " T2: " + \
                       str(x_test.iat[2]) + " HDIST: " + str(x_test.iat[3])

    return formatted_string


# splits the original csv to 2, input and output
def split_csv(csv_data, csv_input, csv_output):
    lines = csv_data.readlines()
    line_number = 0
    for line in lines:

        if line_number == 0:
            inp, outp = line.split(",HDIST,")
            csv_input.write(inp)
            csv_input.write(",HDIST\n")
            csv_output.write(outp)

            line_number += 1
        else:
            lista = line.split(",")

            for i in range(0, 5):
                csv_input.write(lista[i])
                if i < 4:
                    csv_input.write(",")
            csv_input.write("\n")

            for j in range(5, len(lista)):
                csv_output.write(lista[j])
                if j < len(lista) - 1:
                    csv_output.write(",")