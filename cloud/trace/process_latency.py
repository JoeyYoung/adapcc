file = open("latency-hw-g6.4xlarge.4.txt", "r")
outfile = open("latency-hw.txt", "w")

latency_list = []
lines = file.readlines()
for line in lines:
    elems = line.split(" ")
    latency_list.append(float(elems[6].split("=")[1]))

outfile.write(str(latency_list))
file.close()
outfile.close()
