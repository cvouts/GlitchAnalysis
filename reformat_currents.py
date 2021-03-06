# takes the data from the currents_half_time.txt file that was the current output from HSPICE and
# creates the corresponding input and output .csv files.


import os
from common_tools import split_csv

NUMBER_OF_VALUES = 40

# read from .txt and write into .csv
spice_output = open("Data/TXT/currents_half_time.txt", "r")
if os.path.isfile("Data/CSV/Currents/formatted_currents_half.cvs"):
	os.remove("Data/CSV/Currents/formatted_currents_half.csv")
formatted_currents = open("Data/CSV/Currents/formatted_currents_half.csv", "w")

# creating csv headers. For the initial .csv, both the input (C,T1,T2,DIS,HDIST) and the values exist together
i_values = ""
for i in range(1, NUMBER_OF_VALUES+1):
	i_values = i_values + (",I(V6)-" + str(i/2))	 # i/2 for the half step

formatted_currents.write("C,T1,T2,DIS,HDIST")
formatted_currents.write(i_values)
formatted_currents.write("\n")

# below each line of the .txt file is read one by one. Since we know how many values there are in each simulation
# the 'recurring' variable is used to seperate each simulation into different .csv files
lines = spice_output.readlines()
line_number = 1
recurring = 0
for line in lines:
	
	if line_number == 1 or recurring == NUMBER_OF_VALUES+2:  # replacing TOT with a line change
		if recurring == NUMBER_OF_VALUES+2:
			formatted_currents.write("\n")
		line_number += 1
		recurring += 1
		continue
	# removing the line with the current variable names,
	# but connecting what is before and what comes next with ','
	elif line_number == 3 or recurring == NUMBER_OF_VALUES+4:
		formatted_currents.write(",")
		line_number += 1
		recurring = 0
		continue
	
	output = ""
	if line.find("DIS") != -1:	 # finding a line with input data and removing the names

		line = line.replace("C:", "")
		line = line.replace("HDIS:", ",")
		line = line.replace("DIS:", ",")
		line = line.replace("T1:", ",")
		line = line.replace("T2:", ",")
		line = line.replace("\n", "")

		for char in line:  # removing spaces
			if char == " ":
				output = line.replace(char, "")

	if line.find("p") != -1 or line.find("500.00000f") != -1:	 # finding a line with output data
		v6 = line.split()[1]

		# searching for 'n', 'p' and 'm' in order to change the value in uA
		if v6.find("n") != -1:
			v6_string = v6.split("n")[0]
			v6_number = float(v6_string) / 1000
			v6_number = round(v6_number, 7)
			v6 = str(v6_number) 
		elif v6.find("p") != -1:
			v6_string = v6.split("p")[0]
			v6_number = float(v6_string) / (1000*1000)
			v6_number = round(v6_number, 7)
			v6 = str(v6_number)
		elif v6.find("m") != -1:
			v6_string = v6.split("m")[0]
			v6_number = float(v6_string) * 1000
			v6_number = round(v6_number, 7)
			v6 = str(v6_number)
		else:
			v6 = v6.split("u")[0]

		if recurring == NUMBER_OF_VALUES:
			output = v6
		elif recurring < NUMBER_OF_VALUES:
			output = v6 + ","

	if recurring == 0:  # skipping the initial value
		line_number += 1
		recurring += 1
		continue

	if recurring != NUMBER_OF_VALUES+1:
		formatted_currents.write(output)

	line_number += 1
	recurring += 1

spice_output.close()
formatted_currents.close()

# spliting 'formatted_currents_half' into input and output files
csv_data = open("Data/CSV/Currents/formatted_currents_half.csv", "r")
csv_input = open("Data/CSV/Currents/input_currents_half.csv", "w")
csv_output = open("Data/CSV/Currents/output_currents_half.csv", "w")

# calling a method from common_tools, in order to split the original .csv into 2 files, input and output
split_csv(csv_data, csv_input, csv_output)

csv_input.close()
csv_output.close()
csv_data.close()