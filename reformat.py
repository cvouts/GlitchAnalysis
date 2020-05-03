import os

def main():

	f = open("voltages.txt", "r")
	if os.path.isfile("formatted_voltages.cvs"):
		os.remove("formatted_voltages.csv")
	g = open("formatted_voltages.csv", "a+")

	vtimes=""
	for i in range(0,51):
		vtimes = vtimes + (",V3-" + str(i))

	g.write("C,T1,T2,DIS,HDIST")
	g.write(vtimes)
	g.write("\n")

	if f.mode == "r":
		f1 = f.readlines()

		numberOfLines = 1
		recurring = 0

		for line in f1: # for line in list of lines

			if numberOfLines == 1 or recurring == 52: # replacing TOT with a line change
				if recurring == 52:
					g.write("\n")
				numberOfLines += 1
				recurring += 1 
				continue
			elif numberOfLines == 3 or recurring == 54: # removing the line that simply mentions 3 in reference to V3, but 
				g.write(",")							# connecting what is before and what comes next with ','						
				numberOfLines +=1
				recurring = 0
				continue

			if line.find("DIS") != -1: # removing the letters

				line = line.replace("C:", "")
				line = line.replace("DIS:", ",")
				line = line.replace("HDIST:", ",")
				line = line.replace("T1:", ",")
				line = line.replace("T2:", ",")
				line = line.replace("\n", "")

			for char in line: # removing spaces
				if char == " ":
					y = line.replace(char, "")

			if y.find("p") != -1:
				time, voltage = y.split("p")
				if recurring == 50:
					y = voltage
				elif recurring < 50:

					if voltage.find("m") != -1:
						voltage_string, _ = voltage.split("m")
						voltage_number = float(voltage_string) / 1000
						voltage_number = round(voltage_number, 7)
						voltage = str(voltage_number)

					y = voltage + "," # keeping only the voltage values

				y = y.replace("\n", "")

				# if y.find("m") != -1:
				# 	actual, _ = y.split("m")
				# 	number = float(actual) / 1000
				# 	number = round(number, 7)
				# 	actual = str(number)
				# 	#print(actual)
				# 	y = actual

			if recurring == 0: # the initial voltage is not included in the previous if and is also always 1.1000
				y = "1.1000,"

			if recurring != 51: # if this is removed, the final value of every result is going to be doubled, due to the --- line (recurring 51)
				g.write(y)

			numberOfLines += 1
			recurring += 1

	f.close()
	g.close()

if __name__ == "__main__":
	main()