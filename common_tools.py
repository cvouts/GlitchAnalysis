import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV


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


def get_pca(x_in):
    x_standardized = StandardScaler().fit_transform(x_in)
    pca = PCA()
    pca.fit(x_standardized)
    print("PCA variance ratios for all:", pca.explained_variance_ratio_)
    print(pd.DataFrame(pca.components_, columns=x_in.columns))


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


def real_and_predicted_plots(x_test_values, y_actual, y_prediction, ylabel, axis, mae):
    plt.axis(axis)
    time_axis = [0]

    if ylabel == "Voltage (V)":
        upper_limit = 201
        y_actual_with_initial = pd.Series([1.1]).append(y_actual)
        y_prediction_with_initial = pd.Series([1.1]).append(y_prediction)
    else:
        upper_limit = 41
        y_actual_with_initial = pd.Series([0]).append(y_actual)
        y_prediction_with_initial = pd.Series([0]).append(y_prediction)

    for k in range(1, upper_limit):
        if upper_limit == 41:
            time_axis.append(k/2)
        else:
            time_axis.append(k)

    mae_text = "Mean Absolute Error: " + str(mae.round(2)) + "uA"

    actual_line, = plt.plot(time_axis, y_actual_with_initial, "b-")
    predicted_line, = plt.plot(time_axis, y_prediction_with_initial, "r-")
    title = x_test_values
    plt.legend((actual_line, predicted_line), ("actual", "predicted"))
    plt.title(title)
    plt.text(5, -40, mae_text)
    plt.xlabel("Time (ps)")
    plt.ylabel(ylabel)
    # plt.figure()
    plt.show()


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

        if mae < 2:
            print("less than two")
            real_and_predicted_plots(format_x_test_string_data(x_test.iloc[i]), y_test_row,
                                     prediction_row, plot_ylabel, [0, 20, -50, 75], mae)

    print("MAE of less than 2:", round(((less_than_two*100)/y_test.shape[0]), 2), "%\nbetween 2 and 5:",
          round(((two_to_five*100)/y_test.shape[0]), 2), "%\nmore than 5:",
          round(((more_than_five*100)/y_test.shape[0]), 2), "%")
    average_mae = average_mae / y_test.shape[0]
    print("average mean absolute error:", average_mae)


def grid_search(estimator, grid, x, y):
    sc = StandardScaler()
    x = sc.fit_transform(x)

    clf = GridSearchCV(estimator, param_grid=grid, cv=5)
    clf.fit(x, y)
    print(clf.best_params_)
    number_of_estimators = sum(clf.best_params_.values())

    return number_of_estimators


def save_model(model, x_test, y_test, model_path, x_test_path, y_test_path):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    x_test_txt = open(x_test_path, "w")
    y_test_txt = open(y_test_path, "w")
    x_test_txt.write(x_test.to_string())
    y_test_txt.write(y_test.to_string())
    x_test_txt.close()
    y_test_txt.close()


# receives x_test.iloc[i] as parameter, the input data for a single simulation
def format_x_test_string_data(x_test):
    formatted_string = "C: " + str(x_test.iat[0]) + " T1: " + str(x_test.iat[1]) + " T2: " + \
                       str(x_test.iat[2]) + " HDIST: " + str(x_test.iat[3])

    return formatted_string


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