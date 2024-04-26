# read lines from a file
fh = open('gns-split-all.out')
fout = open("gns-split-all.txt", "w")

gns = []
for line in fh:
    info = line.rstrip()
    if "mean gns:" not in info: continue
    elems = info.split("mean gns:")[1]
    gns.append(float(elems))

# write gns list into fout
for g in gns:
    fout.write("%f\n"% g)

fout.close()
fh.close()

