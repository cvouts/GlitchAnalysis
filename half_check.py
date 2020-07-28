# tests whether the current output values with a step of 1 ps and those of a 0.5 ps step are the same
# or if the 0.5 ps step version picks up on spikes in value missed in the 1 ps step version.


import pandas as pd
import matplotlib.pyplot as plt

# the 1 ps step has values up to 200 ps, while the 0.5 ps step has values up to 20 ps
half = pd.read_csv("Data/CSV/Currents/output_currents_half.csv", sep=",")
whole = pd.read_csv("Data/CSV/Currents/output_currents.csv", sep=",")

time_axis = [0]
for k in range(1, 201):
    time_axis.append(k)

time_axis_half = [0]
for k in range(1, 41):
    time_axis_half.append(k/2)

for i in range(0, half.shape[0]):

    # add initial 0 values for clarity
    whole_with_initial = pd.Series([0]).append(whole.iloc[i])
    half_with_initial = pd.Series([0]).append(half.iloc[i])

    # these axes provided a zoomed in view of the difference
    plt.axis([0, 9, -20, 60])
    line_half, = plt.plot(time_axis_half, half_with_initial, "m-^")
    line_whole, = plt.plot(time_axis, whole_with_initial, "b-v")
    plt.legend((line_whole, line_half), ("1ps", "0.5ps"))
    plt.xlabel("Time (ps)")
    plt.ylabel("Current (uA)")
    plt.title("Comparing different time steps")
    plt.show()
