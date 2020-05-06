import pandas as pd
from numpy import mean, std
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import RegressorChain
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedKFold, KFold
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

data_input = pd.read_csv("input_voltages.csv", sep=",")
data_output = pd.read_csv("output_voltages.csv", sep=",")

x = data_input[["C", "T1", "T2", "DIS", "HDIST", "V3-0"]]
y = data_output[["V3-1", "V3-2", "V3-3", "V3-4", "V3-5", "V3-6", "V3-7", "V3-8", "V3-9", "V3-10", "V3-11", "V3-12",
                 "V3-13", "V3-14", "V3-15", "V3-16", "V3-17", "V3-18", "V3-19", "V3-20", "V3-21", "V3-22", "V3-23",
                 "V3-24", "V3-25", "V3-26", "V3-27", "V3-28", "V3-29", "V3-30", "V3-31", "V3-32", "V3-33", "V3-34",
                 "V3-35", "V3-36", "V3-37", "V3-38", "V3-39", "V3-40", "V3-41", "V3-42", "V3-43", "V3-44", "V3-45",
                 "V3-46", "V3-47", "V3-48", "V3-49", "V3-50"]]

params = {"n_neighbors": [2, 3, 4, 5, 6, 7, 8, 9, 10]}
knn = KNeighborsRegressor()

print("KFold")
cv = KFold(n_splits=10, random_state=1)
ne = cross_val_score(KNeighborsRegressor(), x, y, cv=cv)
print("Neighbors:", mean(ne), std(ne))
rf = cross_val_score(RandomForestRegressor(), x, y, cv=cv)
print("Random Forest:", mean(rf), std(rf))

print("Repeated KFold")
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
ne = cross_val_score(KNeighborsRegressor(), x, y, cv=cv)
print("Neighbors:", mean(ne), std(ne))
rf = cross_val_score(RandomForestRegressor(), x, y, cv=cv)
print("Random Forest:", mean(rf), std(rf))


# best_accuracy = 0
# for it in range(10):    # making multiple models in order to eventually save the one with the highest accuracy
#
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
#
#     neighbor_selection_model = GridSearchCV(knn, params, cv=5)  # finding the best number of neighbors
#     neighbor_selection_model.fit(x_train, y_train)
#     neighbor_number = sum(neighbor_selection_model.best_params_.values())
#     model = KNeighborsRegressor(n_neighbors=neighbor_number)
#     model.fit(x_train, y_train)
#
#     # finding accuracy for each model
#     k = 0
#     below50 = 0
#     above80 = 0
#     for i in range(0, len(y_test.count(axis=1))):
#         temp = k
#         for j in range(0, len(y_test.count(axis=0))):
#             if y_test.iat[i, j] == model.predict(x_test)[i, j].round(4):
#                 k += 1
#         print("For instance ", i, " the accuracy was ", round((k-temp)*100/50, 2), "%")
#         if round((k-temp)*100/50, 2) < 50:
#             below50 += 1
#         elif round((k-temp)*100/50, 2) >= 80:
#             above80 += 1
#
#     accuracy = round((k * 100 / (len(y_test.count(axis=1))*len(y_test.count(axis=0)))), 2)
#     print(it, ":", accuracy, "% accuracy, with the number of instances equal or above 80% at", above80,
#           "/343 and below 50% at", below50, "/343")
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#
# print(best_accuracy)
