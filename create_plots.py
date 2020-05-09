import matplotlib.pyplot as plt

DATA_FILE = "output_voltages.csv"
NUMBER_OF_PICOSECONDS = 50

file = open(DATA_FILE, "r")

time_axis = []
for i in range(1, NUMBER_OF_PICOSECONDS+1):
    time_axis.append(i)

lines = file.readlines()
line_number = 1

plt.axis([1, 50, 0.95, 1.12])
plt.title("Output Voltage (V3)")
plt.xlabel("picoseconds")
plt.ylabel("volts")
plt.plot(time_axis, [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1,
                     1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1,
                     1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1], "r--")

for line in lines:
    if line_number == 1:
        line_number += 1
        continue

    data_axis = []
    for word in line.split(","):  # from string to float
        data_axis.append(float(word))

    plt.plot(time_axis, data_axis)
    line_number += 1

print(line_number)
plt.show()
file.close()