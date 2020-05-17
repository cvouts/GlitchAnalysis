def split_csv(csv_data, csv_input, csv_output):
    lines = csv_data.readlines()
    line_number = 0
    for line in lines:

        if line_number == 0:
            inp, outp = line.split(",HDIST,")
            csv_input.write(inp)
            csv_input.write(",HDIST\n")
            csv_output.write(outp)

            line_number += 1
        else:
            lista = line.split(",")

            for i in range(0, 5):
                csv_input.write(lista[i])
                if i < 4:
                    csv_input.write(",")
            csv_input.write("\n")

            for j in range(5, len(lista)):
                csv_output.write(lista[j])
                if j < len(lista) - 1:
                    csv_output.write(",")