import pandas
data = pandas.read_csv("bfslikemcc.csv")
for i in range(130):
    print ("mv " + data['filename'].loc[i] + ".pnml ../ptSelect")
