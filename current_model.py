import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

NUMBER_OF_DATA = 10780
NUMBER_OF_CURRENT_VALUES = 200

data_input = pd.read_csv("Data/CSV/Currents/input_currents.csv", sep=",")
data_output = pd.read_csv("Data/CSV/Currents/output_currents.csv", sep=",")

output_list = []
for i in range(1, NUMBER_OF_CURRENT_VALUES+1):
    output_list.append("I(V6)-" + str(i))

x = data_input[["C", "T1", "T2", "DIS", "HDIST"]]
y = data_output[output_list]

plt.axis([1, 200, -75, 125])
time_axis = []
for i in range(1, 201):
    time_axis.append(i)

accuracy_sum = 0
best_accuracy = 0

model = RandomForestRegressor()
for it in range(10):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    x_standardized = StandardScaler().fit_transform(x)
    pca = PCA()
    pca.fit(x_standardized)
    print("PCA variance ratios for standardized x:", pca.explained_variance_ratio_)

    model.fit(x_train, y_train)

    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)

    train_error = mean_squared_error(y_train, train_prediction)
    test_error = mean_squared_error(y_test, test_prediction)
    print("Standardised x_train and x_test\nRandomForest: train error", train_error, "test error", test_error)


    # finding model accuracy manually
    correct_predictions = 0
    for i in range(0, y_test.shape[0]):  # for each piece of data in the test dataset

        sum_of_previous_predictions = correct_predictions
        for j in range(0, y_test.shape[1]):  # for each voltage value in a test dataset piece
            if y_test.iat[i, j].round(4) == test_prediction[i, j].round(4):  # if actual value == predicted value
                correct_predictions += 1

        plt.plot(time_axis, y_test.iloc[i, :], "b-")
        plt.plot(time_axis, test_prediction[i, :], "r-")
        title = "instance " + str(i+1)
        plt.title(title)
        plt.show()

    accuracy = round((correct_predictions * 100 / y_test.size), 2)  # rule of 3
    print(correct_predictions, "--", accuracy, "% accuracy")

    accuracy_sum += accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
print("average accuracy is", round((accuracy_sum / 10), 2), "%, best accuracy is", best_accuracy, "%")



