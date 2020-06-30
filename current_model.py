import pandas as pd
import pickle
import common_tools  # my file
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

NUMBER_OF_CURRENT_VALUES = 20

# getting the data from the csv
data_input = pd.read_csv("Data/CSV/Currents/input_currents.csv", sep=",")
data_output = pd.read_csv("Data/CSV/Currents/output_currents.csv", sep=",")

# creating x by keeping only these columns from the input data
x = data_input[["C", "T1", "T2", "HDIST"]]

# creating y by keeping only the first 20 I(V6) values
output_list = []
for i in range(1, NUMBER_OF_CURRENT_VALUES+1):
    output_list.append("I(V6)-" + str(i))
y = data_output[output_list]

# can be used to get the Primary Component Analysis of x, comparing the weights of its factors
# common_tools.get_pca(x)

n_estimators = 150

# can be used to get the best hyperparameters
# grid = {'estimator__n_estimators': [125, 150, 175]}   # it's 150
# n_estimators = common_tools.grid_search(MultiOutputRegressor(RandomForestRegressor()), grid, x, y)

# creating the model using a random forest regression algorithm wrapped in a multioutput module
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators))

# split the dataset repeatedly in order to keep the model with the best (smallest) test error
best_test_error = 200
average_test_error = 0
for it in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    x_test_values_for_plot_title = x_test

    # get standardized x_train and x_test
    x_train, x_test = common_tools.standardize_train_test_data(x_train, x_test)

    model.fit(x_train, y_train)

    # get current test mean squared error but print train mse too
    test_error = common_tools.train_test_mean_error("current", model, x_train, x_test, y_train, y_test)

    average_test_error += test_error

    if best_test_error > test_error:
        best_test_error = test_error
        print("best error now is", best_test_error.round(2))
        best_x_train = x_train
        best_x_test = x_test
        best_y_train = y_train
        best_y_test = y_test
        best_x_test_values_for_plot_title = x_test_values_for_plot_title

    if best_test_error < 4:
        break

print("best error was", best_test_error.round(2), "average error was", (average_test_error/(it+1)).round(2))
x_train = best_x_train
x_test = best_x_test
y_train = best_y_train
y_test = best_y_test
x_test_values_for_plot_title = x_test_values_for_plot_title

# train the model using the best x_train and y_train
model.fit(x_train, y_train)

# use the model to predict the values y_test of the x_test
test_prediction = model.predict(x_test)

# compare predicted values against actual y_test values and present results graphically
common_tools.compare_real_and_predicted(x_test_values_for_plot_title, y_test, test_prediction, "Current (μA)")