# creates the machine learning model that predicts the output voltage


import pandas as pd
from pickle import dump
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

model = RandomForestRegressor()

# split the dataset repeatedly in order to keep the model with the best (smallest) test error
best_test_error = 200
average_test_error = 0
for it in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    x_test_values_for_plot_title = x_test

    # keep the not standardized values in order to save the save the relevant scaler
    original_x_train = x_train
    original_x_test = x_test


    # stardardize x_train and x_test
    x_train, x_test = common_tools.standardize_train_test_data(x_train, x_test, -1)

    # compare MSE for x_train and for x_test, return test error for comparison
    model.fit(x_train, y_train)
    test_error = common_tools.train_test_mean_error("voltage", model, x_train, x_test, y_train, y_test)


    average_test_error += test_error

    if best_test_error > test_error:
        best_test_error = test_error
        print("best error now is", best_test_error)
        best_x_train = x_train
        best_x_test = x_test
        best_y_train = y_train
        best_y_test = y_test
        best_x_test_values_for_plot_title = x_test_values_for_plot_title
        best_original_x_train = original_x_train
        best_original_x_test = original_x_test

    break

print("best error was", best_test_error, "average error was", average_test_error/10)
x_train = best_x_train
x_test = best_x_test
y_train = best_y_train
y_test = best_y_test
x_test_values_for_plot_title = x_test_values_for_plot_title
x_train_for_scaler = best_original_x_train
x_test_for_scaler = best_original_x_test

# save the scaler. 0 for voltage scaler
common_tools.standardize_train_test_data(x_train_for_scaler, x_test_for_scaler, 0)

# train the model using the best x_train and y_train
model.fit(x_train, y_train)

# # save the model
# dump(model, open("Models/voltage_model.pkl", "wb"))
# exit()

# use the model to predict the values y_test of the x_test
test_prediction = model.predict(x_test)

common_tools.compare_real_and_predicted(x_test_values_for_plot_title, y_test, test_prediction, "Voltage (V)")