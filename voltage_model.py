import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

NUMBER_OF_DATA = 10780
NUMBER_OF_VOLTAGE_VALUES = 200

data_input = pd.read_csv("Data/CSV/Voltages/input_voltages.csv", sep=",")
data_output = pd.read_csv("Data/CSV/Voltages/output_voltages.csv", sep=",")

output_list = []
for i in range(1, NUMBER_OF_VOLTAGE_VALUES+1):
    output_list.append("V3-" + str(i))

x = data_input[["C", "T1", "T2", "DIS", "HDIST", "V3-0"]]
y = data_output[output_list]

params = {"n_neighbors": [2, 3, 4, 5, 6, 7, 8, 9, 10]}
knn = KNeighborsRegressor()

test_size = 0.1
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

neighbor_selection_model = GridSearchCV(knn, params, cv=5)  # finding the best number of neighbors
neighbor_selection_model.fit(x_train, y_train)
neighbor_number = sum(neighbor_selection_model.best_params_.values())
model = KNeighborsRegressor(n_neighbors=neighbor_number)
model.fit(x_train, y_train)

# finding model accuracy using .score
print(model.score(x_test, y_test))

prediction = model.predict(x_test)

# finding model accuracy manually
correct_predictions = 0
below50 = 0
above80 = 0
test_size = int(NUMBER_OF_DATA * test_size)
voltage_per_data = NUMBER_OF_VOLTAGE_VALUES
for i in range(0, test_size):   # for each piece of data in the test dataset
    sum_of_previous_predictions = correct_predictions
    for j in range(0, voltage_per_data):  # for each voltage value in a test dataset piece
        if y_test.iat[i, j] == prediction[i, j].round(4):  # if actual value == predicted value
            correct_predictions += 1

    print("For instance", i+1, "the accuracy was ",
          round((correct_predictions-sum_of_previous_predictions)*100/voltage_per_data, 2), "%")  # rule of 3

    if round((correct_predictions-sum_of_previous_predictions)*100/voltage_per_data, 2) < 50:
        below50 += 1
    elif round((correct_predictions-sum_of_previous_predictions)*100/voltage_per_data, 2) >= 80:
        above80 += 1

accuracy = round((correct_predictions * 100 / (voltage_per_data*test_size)), 2)  # rule of 3
print(accuracy, "% accuracy, with the number of instances equal or above 80% at", above80,
      "/1078 and below 50% at", below50, "/1078")
