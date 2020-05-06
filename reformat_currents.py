import os 

NUMBER_OF_VALUES = 500

spice_output = open("asdf.txt", "r")
if os.path.isfile("formatted_currents.cvs"):
	os.remove("formatted_currents.csv")
formatted_currents = open("formatted_currents.csv", "w")

# creating csv headers
i_values = ""
for i in range(0,NUMBER_OF_VALUES+1):
	i_values = i_values + (",I(V6)-" + str(i))

formatted_currents.write("C,T1,T2,DIS,HDIST")
formatted_currents.write(i_values)
formatted_currents.write("\n")

lines = spice_output.readlines()

line_number = 1
recurring = 0

for line in lines:
	
	if line_number == 1 or recurring == NUMBER_OF_VALUES+2:
		if recurring == NUMBER_OF_VALUES+2:
			formatted_currents.write("\n")
		line_number += 1
		recurring += 1
		continue
	elif line_number == 3 or recurring == NUMBER_OF_VALUES+4:
		formatted_currents.write(",")
		line_number += 1
		recurring = 0
		continue
	
	output = ""
	if line.find("DIS") != -1:

		line = line.replace("C:", "")
		line = line.replace("HDIS:", ",")
		line = line.replace("DIS:", ",")
		line = line.replace("T1:", ",")
		line = line.replace("T2:", ",")
		line = line.replace("\n", "")

		for char in line: # removing spaces
			if char == " ":
				output = line.replace(char, "")

	if line.find("p") != -1:							
		v6 = line.split()[1]					
												
		if v6.find("n") != -1:
			v6_string = v6.split("n")[0]
			v6_number = float(v6_string) / 1000
			v6_number = round(v6_number, 7)
			v6 = str(v6_number) 
		else:
			v6 = v6.split("u")[0]

		if recurring == NUMBER_OF_VALUES:
			output = v6
		elif recurring < NUMBER_OF_VALUES:

			output = v6 + ","

	if recurring == 0:
		output = "2.7139,"

	if recurring != NUMBER_OF_VALUES+1:
		formatted_currents.write(output)

	line_number += 1
	recurring += 1

spice_output.close()
formatted_currents.close()

# Spliting 'formatted_currents' into input and output files - THE CSV NEED TO BE WRITTEN IN BOTH FILES
formatted_currents = open("formatted_currents.csv", "r")
input_current = open("input_current.csv", "w")
output_current = open("output_current.csv", "w")

formatted_lines = formatted_currents.readlines()
for line in formatted_lines:

	if line.find(",2.7139,") == -1:
		print(line)
		inp, outp = line.split(",I(V6)-0,")
		input_current.write(inp)
		input_current.write(",I(V6)-0\n")
		output_current.write(outp)
	else:
		inp, outp = line.split(",2.7139,")

		input_current.write(inp)
		input_current.write(",2.7139\n")
		output_current.write(outp)

input_current.close()
output_current.close()
formatted_currents.close()