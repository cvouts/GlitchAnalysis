import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

NUMBER_OF_DATA = 10780
NUMBER_OF_VOLTAGE_VALUES = 200

data_input = pd.read_csv("Data/CSV/Voltages/input_voltages.csv", sep=",")
data_output = pd.read_csv("Data/CSV/Voltages/output_voltages.csv", sep=",")

output_list = []
for i in range(1, NUMBER_OF_VOLTAGE_VALUES+1):
    output_list.append("V3-" + str(i))

time_axis = []
for i in range(1, 201):
    time_axis.append(i)

x = data_input[["C", "T1", "T2", "HDIST"]]
y = data_output[output_list]

accuracy_sum = 0
best_accuracy = 0
model = RandomForestRegressor()

best_test_error = 200

# for it in range(10):
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(x, y):
    x_train = x.iloc[train_index]
    y_train = y.iloc[train_index]
    x_test = x.iloc[test_index]
    y_test = y.iloc[test_index]

    # stardardize x
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # compare MSE for x_train and for x_test
    model.fit(x_train, y_train)
    test_prediction = model.predict(x_test)
    train_prediction = model.predict(x_train)
    train_error = mean_squared_error(y_train, train_prediction)
    test_error = mean_squared_error(y_test, test_prediction)
    print("train error", train_error, "test error", test_error)

    # if it == 0:
    #     with open("Models/voltage_model", "wb") as f:
    #         pickle.dump(model, f)
    #     model_test_data_x = open("Models/model_test_data_x.txt", "w")
    #     model_test_data_y = open("Models/model_test_data_y.txt", "w")
    #     model_test_data_x.write(x_test.to_string())
    #     model_test_data_y.write(y_test.to_string())
    #     model_test_data_x.close()
    #     model_test_data_y.close()

    if best_test_error > test_error:
        best_test_error = test_error
        print("best error now is", best_test_error.round(2))
        best_x_train = x_train
        best_x_test = x_test
        best_y_train = y_train
        best_y_test = y_test

print("best error was", best_test_error.round(2))
x_train = best_x_train
x_test = best_x_test
y_train = best_y_train
y_test = best_y_test

model.fit(x_train, y_train)
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
        plt.axis([1, 200, 0.95, 1.12])
        actual_line, = plt.plot(time_axis, y_test.iloc[i, :], "b-")
        predicted_line, = plt.plot(time_axis, test_prediction[i, :], "r-")
        title = "instance " + str(i + 1)
        plt.legend((actual_line, predicted_line), ("actual", "predicted"))
        plt.title(title)
        plt.xlabel("picosecond")
        plt.ylabel("volt")
        plt.show()
