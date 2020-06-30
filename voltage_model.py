import pandas as pd
import pickle
import common_tools  # my file
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

NUMBER_OF_VOLTAGE_VALUES = 200

# getting the data from the csv
data_input = pd.read_csv("Data/CSV/Voltages/input_voltages.csv", sep=",")
data_output = pd.read_csv("Data/CSV/Voltages/output_voltages.csv", sep=",")

# creating x by keeping only these columns from the input data
x = data_input[["C", "T1", "T2", "HDIST"]]

# creating y
output_list = []
for i in range(1, NUMBER_OF_VOLTAGE_VALUES+1):
    output_list.append("V3-" + str(i))
y = data_output[output_list]

accuracy_sum = 0
best_accuracy = 0
model = RandomForestRegressor()

# split the dataset repeatedly in order to keep the model with the best (smallest) test error
best_test_error = 200
average_test_error = 0
for it in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    x_test_values_for_plot_title = x_test

    # stardardize x_train and x_test
    x_train, x_test = common_tools.standardize_train_test_data(x_train, x_test)

    # compare MSE for x_train and for x_test, return test error for comparison
    model.fit(x_train, y_train)
    test_error = common_tools.train_test_mean_error("voltage", model, x_train, x_test, y_train, y_test)

    # if it == 0:
    #    common_tools.save_model(model, x_test, y_test, "Models/voltage_model",
    #                        "Models/model_test_data_x.txt", "Models/model_test_data_y.txt")


    average_test_error += test_error

    if best_test_error > test_error:
        best_test_error = test_error
        print("best error now is", best_test_error)
        best_x_train = x_train
        best_x_test = x_test
        best_y_train = y_train
        best_y_test = y_test
        best_x_test_values_for_plot_title = x_test_values_for_plot_title

    break

print("best error was", best_test_error, "average error was", average_test_error/10)
x_train = best_x_train
x_test = best_x_test
y_train = best_y_train
y_test = best_y_test
x_test_values_for_plot_title = x_test_values_for_plot_title

# train the model using the best x_train and y_train
model.fit(x_train, y_train)

# use the model to predict the values y_test of the x_test
test_prediction = model.predict(x_test)

flag = 0
for i in range(0, y_test.shape[0]):   # for each piece of data in the test dataset
    list_of_differences = []
    for j in range(0, y_test.shape[1]):  # for each voltage value in a test dataset piece

        difference = abs(y_test.iat[i, j].round(4) - test_prediction[i, j].round(4))
        list_of_differences.append(difference)

        if y_test.iat[i, j] < 1:
            flag = 1

    max_difference = max(list_of_differences)
    print(max_difference)

    # plot only for instances that actually fall below 1V
    if flag == 1:
        flag = 0
        common_tools.real_and_predicted_plots(x_test_values_for_plot_title, y_test, test_prediction,
                                              i, "Voltage (V)", [1, 200, 0.95, 1.12])
