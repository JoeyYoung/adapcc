# read lines from a file
fh = open('nohup.out')
fout = open("accuracy_bc32_4g_vgg16_july4.txt", "w")

accuracy = []
for line in fh:
    info = line.rstrip()
    if "Acc@1" not in info: continue
    elems = info.split("Acc@1")[1]
    temp = elems.split("(")
    accuracy.append(float(temp[0]))

# write accuracy list into fout
for acc in accuracy:
    fout.write("%f\n"% acc)

fout.close()
fh.close()

