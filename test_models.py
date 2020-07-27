import sys
from pickle import load

# voltage_model = load(open("Models/voltage_model.pkl", "rb"))
# voltage_scaler = load(open("Models/voltage_scaler.pkl", "rb"))

current_model = load(open("Models/current_model.pkl", "rb"))
current_scaler = load(open("Models/current_scaler.pkl", "rb"))

input_values = [[sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]]]

x_predict = current_scaler.transform(input_values)
prediction = current_model.predict(x_predict)


for i in range(0, prediction.size):
    print((i+1)/2, prediction[0][i])
