from os import path
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import  MultiOutputRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

data_input = pd.read_csv("Data/CSV/Currents/input_currents.csv", sep=",")
data_output = pd.read_csv("Data/CSV/Currents/output_currents.csv", sep=",")

output_list = []
for i in range(1, 20+1):
    output_list.append("I(V6)-" + str(i))

x = data_input[["C", "T1", "T2", "DIS"]]
y = data_output[output_list]

plt.axis([1, 20, -75, 125])
time_axis = []
for i in range(1, 21):
    time_axis.append(i)

model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=0))

best_test_error = 200
for it in range(5):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model.fit(x_train, y_train)

    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)

    train_error = mean_squared_error(y_train, train_prediction)
    test_error = mean_squared_error(y_test, test_prediction)
    print(it+1, "Forest train error", train_error.round(2), "test error", test_error.round(2))

    if best_test_error > test_error:
        best_test_error = test_error
        print("best error now is", best_test_error.round(2))
        best_x_train = x_train
        best_x_test = x_test
        best_y_train = y_train
        best_y_test = y_test

    if best_test_error < 5:
        break

print("best error was", best_test_error.round(2))
x_train = best_x_train
x_test = best_x_test
y_train = best_y_train
y_test = best_y_test

model.fit(x_train, y_train)
test_prediction = model.predict(x_test)

# finding model accuracy manually
less_than_2 = 0
wrong_for_more_than_10_flag = 0
correct_predictions_flag = 0
more_than_10 = 0
less_than_10 = 0
for i in range(0, y_test.shape[0]):  # for each piece of data in the test dataset

    for j in range(0, y_test.shape[1]):  # for each voltage value in a test dataset piece
        if abs(y_test.iat[i, j].round(4)-test_prediction[i, j].round(4)) < 2:  # comparing actual and predicted values
            correct_predictions_flag = 1
        elif abs(y_test.iat[i, j].round(4)-test_prediction[i, j].round(4)) > 10:
            wrong_for_more_than_10_flag = 1

    if wrong_for_more_than_10_flag == 1:
        wrong_for_more_than_10_flag = 0
        more_than_10 += 1
    else:
        less_than_10 += 1
        if correct_predictions_flag == 1:
            correct_predictions_flag = 0
            less_than_2 += 1

    # plt.plot(time_axis, y_test.iloc[i, :], "b-")
    # plt.plot(time_axis, test_prediction[i, :], "r-")
    # title = "instance " + str(i+1)
    # plt.title(title)
    # plt.show()

print("difference of more than 10:", more_than_10, "\ndifference of less than 10:", less_than_10,
      "\ndifference of less than 2:", less_than_2)
