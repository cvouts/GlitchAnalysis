# plots every simulation (10 at a time) from the dataset onto the same graph. It takes inline arguments in this fashion:
# path-to-output-csv-file max-number-of-values-in-file number-of-values-needed. So for example:
# "Data/CSV/Voltages/output_voltages.csv" 200 100
# will print the first 100 values of the voltage values in the dataset.


import matplotlib.pyplot as plt
import sys

DATA_FILE = sys.argv[1]  # path to output csv file
NUMBER_OF_PICOSECONDS = int(sys.argv[2])  # max number of output values
PLOT_TIME_LENGTH = int(sys.argv[3])  # number of values to be plotted

if sys.argv[1].find("Currents") != -1:  # Setup for Currents
    INITIAL_VALUE = 2.7139 / 1000
    y_axis_max = 125
    y_axis_min = -75
    title = "Power Supply Current I(V6)"
    ylabel = "microampere"
else:                                   # Setup for Voltages
    INITIAL_VALUE = 1.1000
    y_axis_max = 1.12
    y_axis_min = 0.95
    title = "Output Voltage (V3)"
    ylabel = "volt"

file = open(DATA_FILE, "r")

time_axis = []
initial_value_line = []
for i in range(1, NUMBER_OF_PICOSECONDS+1):
    time_axis.append(i)
    initial_value_line.append(INITIAL_VALUE)

lines = file.readlines()
line_number = 0

plt.axis([1, 25, y_axis_min, y_axis_max])
plt.title(title)
plt.xlabel("picosecond")
plt.ylabel(ylabel)
plt.plot(time_axis, initial_value_line, "r--")

per10 = 0  # the value used to stop every 10 plots

for line in lines:
    if line_number == 0:
        line_number += 1
        continue

    per10 += 1

    data_axis = []
    for word in line.split(","):  # from string to float
        data_axis.append(float(word))

    plt.plot(time_axis, data_axis)

    if per10 == 10:
        print("from", line_number-10, "to", line_number)
        plt.show()
        per10 = 0

    line_number += 1

plt.show()
file.close()