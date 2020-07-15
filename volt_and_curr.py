import pandas as pd
import matplotlib.pyplot as plt
import common_tools

x = pd.read_csv("Data/CSV/Currents/input_currents.csv")
y_c = pd.read_csv("Data/CSV/Currents/output_currents_half.csv")
y_v = pd.read_csv("Data/CSV/Voltages/output_voltages.csv")

x_common = x[["C", "T1", "T2", "HDIST"]]

current_list = []
axis_current = [0]
for i in range(1, 41):
    axis_current.append(i/2)
    current_list.append("I(V6)-" + str(i/2))
y_current = y_c[current_list]

voltage_list = []
axis_voltage = [0]
for i in range(1, 101):
    voltage_list.append("V3-" + str(i))
    axis_voltage.append(i)
y_voltage = y_v[voltage_list]

for i in range(0, x_common.shape[0]):

    this_x = common_tools.format_x_test_string_data(x_common.iloc[i])

    print(this_x)

    y_vvv = pd.Series([1.1]).append(y_voltage.iloc[i])
    y_ccc = pd.Series([0.0027139]).append(y_current.iloc[i]) # initial current value is 2.7139n, so 0.0027139u

    plt.plot(axis_voltage, y_vvv, "b-")
    plt.title(this_x)
    plt.legend("voltage")
    plt.xlabel("Time (pS)")
    plt.ylabel("Voltage V (V)")
    plt.show()

    plt.plot(axis_current, y_ccc, "g-")
    plt.title(this_x)
    plt.legend("current")
    plt.xlabel("Time (pS)")
    plt.ylabel("Current I (uA)")
    plt.show()