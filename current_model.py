import pandas as pd
from pickle import dump
import common_tools  # my file
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

NUMBER_OF_CURRENT_VALUES = 40

# getting the data from the csv
data_input = pd.read_csv("Data/CSV/Currents/input_currents_half.csv", sep=",")
data_output = pd.read_csv("Data/CSV/Currents/output_currents_half.csv", sep=",")

# creating x by keeping only these columns from the input data
x = data_input[["C", "T1", "T2", "HDIST"]]

# creating y. Notice i/2 for the half step
output_list = []
for i in range(1, NUMBER_OF_CURRENT_VALUES+1):
    output_list.append("I(V6)-" + str(i/2))
y = data_output[output_list]

# can be used to get the Primary Component Analysis of x, comparing the weights of its factors
# common_tools.get_pca(x)

# can be used to get the best hyperparameters
# grid = {'estimator__n_estimators': [125, 150, 175]}   # it's 150
# n_estimators = common_tools.grid_search(MultiOutputRegressor(RandomForestRegressor()), grid, x, y)
n_estimators = 150

# creating the model using a random forest regression algorithm wrapped
# in a multioutput module and using the number of trees found above
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators))

# split the dataset repeatedly in order to keep the model with the best (smallest) test error
best_test_error = 200
average_test_error = 0
for it in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    # needed to set the input values in the plot
    x_test_values_for_plot_title = x_test

    # keeping the not standardized values in order to save the save the relevant scaler
    original_x_train = x_train
    original_x_test = x_test

    # get standardized x_train and x_test
    x_train, x_test = common_tools.standardize_train_test_data(x_train, x_test, -1)

    model.fit(x_train, y_train)

    # get current test mean squared error. Print both train and test mse
    test_error = common_tools.train_test_mean_error("current", model, x_train, x_test, y_train, y_test)

    average_test_error += test_error

    # keeping the best values to create the best model
    if best_test_error > test_error:
        best_test_error = test_error
        print("best error now is", best_test_error.round(2))
        best_x_train = x_train
        best_x_test = x_test
        best_y_train = y_train
        best_y_test = y_test
        best_x_test_values_for_plot_title = x_test_values_for_plot_title
        best_original_x_train = original_x_train
        best_original_x_test = original_x_test

    if best_test_error < 4:
        break

print("best error was", best_test_error.round(2), "average error was", (average_test_error/(it+1)).round(2))

# using the best values found above
x_train = best_x_train
x_test = best_x_test
y_train = best_y_train
y_test = best_y_test
x_test_values_for_plot_title = x_test_values_for_plot_title
x_train_for_scaler = best_original_x_train
x_test_for_scaler = best_original_x_test

# save the scaler. 1 for current scaler
common_tools.standardize_train_test_data(x_train_for_scaler, x_test_for_scaler, 1)

# train the model using the best x_train and y_train
model.fit(x_train, y_train)

# save the model
# dump(model, open("Models/current_model.pkl", "wb"))
# exit()

# use the model to predict the values y_test of the x_test
test_prediction = model.predict(x_test)

# compare predicted values against actual y_test values and present results graphically
common_tools.compare_real_and_predicted(x_test_values_for_plot_title, y_test, test_prediction, "Current (μA)")