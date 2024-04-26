file = open("bandwidth-hw-g6.4xlarge.4.txt", "r")
outfile = open("bandwidth-hw.txt", "w")

bands = []
lines = file.readlines()
for line in lines:
    elems = line.split(" ")
    if elems[11] == "": continue
    bands.append(float(elems[11]))

outfile.write(str(bands))
file.close()
outfile.close()
