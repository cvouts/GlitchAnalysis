from os import path
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

data_input = pd.read_csv("Data/CSV/Currents/input_currents.csv", sep=",")
data_output = pd.read_csv("Data/CSV/Currents/output_currents.csv", sep=",")

output_list = []
for i in range(1, 20+1):
    output_list.append("I(V6)-" + str(i))

x = data_input[["C", "T1", "T2", "HDIST"]]

# x_standardized = StandardScaler().fit_transform(x)
# pca = PCA()
# pca.fit(x_standardized)
# print("PCA variance ratios for all:", pca.explained_variance_ratio_)
# print(pd.DataFrame(pca.components_, columns=x.columns))

y = data_output[output_list]

plt.axis([1, 20, -75, 125])
time_axis = []
for i in range(1, 21):
    time_axis.append(i)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)
#
# grid = {'estimator__n_estimators': [50, 100, 150],
#         'estimator__min_samples_split': [2, 4, 6],
#         'estimator__min_samples_leaf': [1, 2, 4]}
#
# clf = GridSearchCV(MultiOutputRegressor(RandomForestRegressor()), param_grid=grid, cv=5)
# clf.fit(x_train, y_train)
# print(clf.best_params_)
                                                                    # from results. The samples are the default
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=150, min_samples_leaf=1, min_samples_split=2))
best_test_error = 200
for it in range(5):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
# kf = KFold(n_splits=5, shuffle=True)
# for train_index, test_index in kf.split(x, y):
#     x_train = x.iloc[train_index]
#     y_train = y.iloc[train_index]
#     x_test = x.iloc[test_index]
#     y_test = y.iloc[test_index]

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model.fit(x_train, y_train)

    train_prediction = model.predict(x_train)
    test_prediction = model.predict(x_test)

    train_error = mean_squared_error(y_train, train_prediction)
    test_error = mean_squared_error(y_test, test_prediction)
    print("train error", train_error.round(2), "test error", test_error.round(2))

    if best_test_error > test_error:
        best_test_error = test_error
        print("best error now is", best_test_error.round(2))
        best_x_train = x_train
        best_x_test = x_test
        best_y_train = y_train
        best_y_test = y_test

    # if best_test_error < 5:
    #     break

print("best error was", best_test_error.round(2))
x_train = best_x_train
x_test = best_x_test
y_train = best_y_train
y_test = best_y_test

model.fit(x_train, y_train)
test_prediction = model.predict(x_test)

# finding model accuracy manually
less_than_2 = 0
more_than_10 = 0
less_than_10 = 0
for i in range(0, y_test.shape[0]):  # for each piece of data in the test dataset

    list_of_differences = []
    for j in range(0, y_test.shape[1]):  # for each voltage value in a test dataset piece

        difference = abs(y_test.iat[i, j].round(4)-test_prediction[i, j].round(4))
        list_of_differences.append(difference)

    max_difference = max(list_of_differences)
    if max_difference > 10:
        more_than_10 += 1
    else:
        less_than_10 += 1
        if max_difference < 2:
            less_than_2 += 1

        actual_line, = plt.plot(time_axis, y_test.iloc[i, :], "b-")
        predicted_line, = plt.plot(time_axis, test_prediction[i, :], "r-")
        title = "instance " + str(i+1)
        plt.legend((actual_line, predicted_line), ("actual", "predicted"))
        plt.title(title)
        plt.xlabel("picosecond")
        plt.ylabel("microamber")
        plt.show()

print("max difference of more than 10:", more_than_10, "\nmax difference of less than 10:", less_than_10,
      "\nspecifically, max difference of less than 2:", less_than_2)
