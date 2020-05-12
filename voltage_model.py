import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

NUMBER_OF_VALUES = 200

data_input = pd.read_csv("Data/CSV/Voltages/input_voltages.csv", sep=",")
data_output = pd.read_csv("Data/CSV/Voltages/output_voltages.csv", sep=",")

output_list = []
for i in range(1, NUMBER_OF_VALUES+1):
    output_list.append("V3-" + str(i))

x = data_input[["C", "T1", "T2", "DIS", "HDIST", "V3-0"]]
y = data_output[output_list]

params = {"n_neighbors": [2, 3, 4, 5, 6, 7, 8, 9, 10]}
knn = KNeighborsRegressor()

# print("KFold")
# cv = KFold(n_splits=10, random_state=1)
# ne = cross_val_score(KNeighborsRegressor(n_neighbors=neighbor_number), x, y, scoring='neg_mean_absolute_error', cv=cv,
#                      n_jobs=-1, error_score='raise')
# print("Neighbors:", mean(ne), std(ne))
#
# print("Repeated KFold")
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# ne = cross_val_score(KNeighborsRegressor(n_neighbors=neighbor_number), x, y, scoring='neg_mean_absolute_error', cv=cv,
#                      n_jobs=-1, error_score='raise')
# print("Neighbors:", mean(ne), std(ne))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

neighbor_selection_model = GridSearchCV(knn, params, cv=5)  # finding the best number of neighbors
neighbor_selection_model.fit(x_train, y_train)
neighbor_number = sum(neighbor_selection_model.best_params_.values())
model = KNeighborsRegressor(n_neighbors=neighbor_number)
model.fit(x_train, y_train)

# finding model accuracy using .score
print(model.score(x_test, y_test))

# finding model accuracy manually
k = 0
below50 = 0
above80 = 0
for i in range(0, len(y_test.count(axis=1))):
    temp = k
    for j in range(0, len(y_test.count(axis=0))):
        if y_test.iat[i, j] == model.predict(x_test)[i, j].round(4):
            k += 1
    print("For instance", i+1, "the accuracy was ", round((k-temp)*100/NUMBER_OF_VALUES, 2), "%")
    if round((k-temp)*100/NUMBER_OF_VALUES, 2) < 50:
        below50 += 1
    elif round((k-temp)*100/NUMBER_OF_VALUES, 2) >= 80:
        above80 += 1

accuracy = round((k * 100 / (len(y_test.count(axis=1))*len(y_test.count(axis=0)))), 2)
print(accuracy, "% accuracy, with the number of instances equal or above 80% at", above80,
      "/1078 and below 50% at", below50, "/1078")
