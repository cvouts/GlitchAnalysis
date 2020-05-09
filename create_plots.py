import matplotlib.pyplot as plt
import sys

DATA_FILE = "Data/CSV/"+sys.argv[1]  # Data/CSV/KIND/some output file
NUMBER_OF_PICOSECONDS = int(sys.argv[2])  # up to 500
PLOT_TIME_LENGTH = int(sys.argv[3])
if sys.argv[1].find("Currents") != -1:  # Setup for Currents
    INITIAL_VALUE = 2.7139
    y_axis_max = 125
    y_axis_min = -75
    title = "Power Supply Current I(V6)"
    ylabel = "microamber"
else:                                   # Setup for Voltages
    INITIAL_VALUE = 1.1000
    y_axis_max = 1.12
    y_axis_min = 0.9
    title = "Output Voltage (V3)"
    ylabel = "volt"

file = open(DATA_FILE, "r")

time_axis = []
initial_value_line = []
for i in range(1, NUMBER_OF_PICOSECONDS+1):
    time_axis.append(i)
    initial_value_line.append(INITIAL_VALUE)

lines = file.readlines()
line_number = 1

plt.axis([1, PLOT_TIME_LENGTH, y_axis_min, y_axis_max])
plt.title(title)
plt.xlabel("picosecond")
plt.ylabel(ylabel)
plt.plot(time_axis, initial_value_line, "r--")

for line in lines:
    if line_number == 1:
        line_number += 1
        continue

    data_axis = []
    for word in line.split(","):  # from string to float
        data_axis.append(float(word))

    plt.plot(time_axis, data_axis)

plt.show()
file.close()