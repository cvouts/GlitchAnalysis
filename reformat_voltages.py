import os
from split_formatted_csv import split_csv

NUMBER_OF_VALUES = 200

text_data = open("Data/TXT/voltages.txt", "r")
if os.path.isfile("Data/CSV/formatted_voltages.cvs"):
	os.remove("Data/CSV/Voltages/formatted_voltages.csv")
	os.remove("Data/CSV/Voltages/input_voltages.csv")
	os.remove("Data/CSV/Voltages/output_voltages.csv")
csv_data = open("Data/CSV/Voltages/formatted_voltages.csv", "w")

vtimes = ""
for i in range(1, NUMBER_OF_VALUES+1):
	vtimes = vtimes + (",V3-" + str(i))

csv_data.write("C,T1,T2,DIS,HDIST")
csv_data.write(vtimes)
csv_data.write("\n")

all_lines = text_data.readlines()

line_number = 1
recurring = 0

for line in all_lines:  # for line in list of lines

	if line_number == 1 or recurring == NUMBER_OF_VALUES+2:  # replacing TOT with a line change
		if recurring == NUMBER_OF_VALUES+2:
			csv_data.write("\n")
		line_number += 1
		recurring += 1
		continue
	elif line_number == 3 or recurring == NUMBER_OF_VALUES+4: # removing the line that simply mentions 3 in reference to V3, but
		csv_data.write(",")							# connecting what is before and what comes next with ','
		line_number += 1
		recurring = 0
		continue

	if line.find("DIS") != -1:  # removing the letters

		line = line.replace("C:", "")
		line = line.replace("DIS:", ",")
		line = line.replace("HDIST:", ",")
		line = line.replace("T1:", ",")
		line = line.replace("T2:", ",")
		line = line.replace("\n", "")

	output = line
	for char in line:  # removing spaces
		if char == " ":
			output = line.replace(char, "")

	if output.find("p") != -1:
		time, voltage = output.split("p")
		if recurring == NUMBER_OF_VALUES:
			output = voltage
		elif recurring < NUMBER_OF_VALUES:

			if voltage.find("m") != -1:
				voltage_string, _ = voltage.split("m")
				voltage_number = float(voltage_string) / 1000
				voltage_number = round(voltage_number, 7)
				voltage = str(voltage_number)

			output = voltage + "," # keeping only the voltage values

		output = output.replace("\n", "")

	if recurring == 0:  # the initial voltage is not included in the previous if and is also always 1.1000
		line_number += 1
		recurring += 1
		continue

	if recurring != NUMBER_OF_VALUES+1: # if this is removed, the final value of every result is going to be doubled, due to the --- line (recurring 51)
		csv_data.write(output)

	line_number += 1
	recurring += 1

text_data.close()
csv_data.close()

csv_data = open("Data/CSV/Voltages/formatted_voltages.csv", "r")
csv_input = open("Data/CSV/Voltages/input_voltages.csv", "w")
csv_output = open("Data/CSV/Voltages/output_voltages.csv", "w")

split_csv(csv_data, csv_input, csv_output)

csv_input.close()
csv_output.close()
csv_data.close()