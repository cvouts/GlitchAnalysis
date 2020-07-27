import pandas as pd
import matplotlib.pyplot as plt

half = pd.read_csv("Data/CSV/Currents/output_currents_half.csv", sep=",")
whole = pd.read_csv("Data/CSV/Currents/output_currents.csv", sep=",")

time_axis = [0]
for k in range(1, 201):
    time_axis.append(k)

time_axis_half = [0]
for k in range(1, 41):
    time_axis_half.append(k/2)

for i in range(0, half.shape[0]):

    whole_with_initial = pd.Series([0]).append(whole.iloc[i])
    half_with_initial = pd.Series([0]).append(half.iloc[i])

    print(i)
    plt.axis([0, 20, -50, 90])
    line_whole, = plt.plot(time_axis, whole_with_initial, "b-")

    line_half, = plt.plot(time_axis_half, half_with_initial, "m-")
    plt.legend((line_whole, line_half), ("1ps", "0.5ps"))
    plt.show()
