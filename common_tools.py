import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


def standardize_train_test_data(x_train, x_test):
    sc = StandardScaler()
    standard_x_train = sc.fit_transform(x_train)
    standard_x_test = sc.transform(x_test)

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


def real_and_predicted_plots(x_test_values, y_test, test_predict, current_element, ylabel, axis):
    plt.axis(axis)
    time_axis = []

    if ylabel == "Voltage (V)":
        upper_limit = 201
    else:
        upper_limit = 21

    for k in range(1, upper_limit):
        time_axis.append(k)

    actual_line, = plt.plot(time_axis, y_test.iloc[current_element, :], "b-")
    predicted_line, = plt.plot(time_axis, test_predict[current_element, :], "r-")
    # title = "instance " + str(current_element+1)
    title = x_test_values
    plt.legend((actual_line, predicted_line), ("actual", "predicted"))
    plt.title(title)
    plt.xlabel("Time (pS)")
    plt.ylabel(ylabel)
    # plt.figure()
    plt.show()


def compare_real_and_predicted(x_test, y_test, test_prediction, plot_ylabel):
    less_than_2 = 0
    more_than_10 = 0
    less_than_10 = 0
    for i in range(0, y_test.shape[0]):  # for each piece of data in the test dataset

        list_of_differences = []
        for j in range(0, y_test.shape[1]):  # for each voltage/current value in a test dataset piece

            difference = abs(y_test.iat[i, j].round(4) - test_prediction[i, j].round(4))
            list_of_differences.append(difference)

        max_difference = max(list_of_differences)
        if max_difference > 10:
            more_than_10 += 1

            real_and_predicted_plots(format_x_test_string_data(x_test.iloc[i]), y_test,
                                     test_prediction, i, plot_ylabel, [1, 20, -50, 75])
        else:
            less_than_10 += 1
            if max_difference < 2:
                less_than_2 += 1

    print("max difference of more than 10:", more_than_10, "\nmax difference of less than 10:", less_than_10,
          "\nspecifically, max difference of less than 2:", less_than_2)


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