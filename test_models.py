# loads the models and scalers saved by voltage_model and current_model and uses them on
# input provided as inline arguments. It produces two vectors of predictions.
# The input looks like: path-to-folder-containing-models-and-scalers C T1 T2 HDIST. So for example:
# Models 0.365616 0.00117378 0.00117378 0.2474351375

import sys
from pickle import load

voltage_model = load(open(sys.argv[1]+"/voltage_model.pkl", "rb"))
voltage_scaler = load(open(sys.argv[1]+"/voltage_scaler.pkl", "rb"))

current_model = load(open(sys.argv[1]+"/current_model.pkl", "rb"))
current_scaler = load(open(sys.argv[1]+"/current_scaler.pkl", "rb"))

input_values = [[sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]]]

x_predict_current = current_scaler.transform(input_values)
current_prediction = current_model.predict(x_predict_current)

x_predict_voltage = voltage_scaler.transform(input_values)
voltage_prediction = voltage_model.predict(x_predict_voltage)

print("Power supply current values (in uA) every 0.5 ps, from 0.5 to 20 ps\n", current_prediction[0])
print("\nOutput voltage values (in V) every 1 ps from 1 to 100 ps\n", voltage_prediction[0][:100])


