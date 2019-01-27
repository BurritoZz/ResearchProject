import pandas

data = pandas.read_csv("rest.csv")
data = data[data.status != 'ootime']
data = data[data.status != 'error"boostperm']
data = data[data.status != 'error"UnknownError']
data = data[data.status != 'error']

data.to_csv("rest.csv")

print(data)
