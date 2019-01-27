import pandas
import matplotlib

df = pandas.read_csv("bfsprevloopmcc.csv")
df = df.drop(labels='satgranularity', axis=1)
df = df.dropna()
df = df.reset_index()

var = 'profile'
data = pandas.concat([df['timeAvg'], df[var]], axis=1)
plot = data.plot.scatter(x=var, y='timeAvg')
fig = plot.get_figure()
fig.savefig("profilePlot")
var = 'span'
data = pandas.concat([df['timeAvg'], df[var]], axis=1)
plot = data.plot.scatter(x=var, y='timeAvg')
fig = plot.get_figure()
fig.savefig("spanPlot")
var = 'Transitions'
data = pandas.concat([df['timeAvg'], df[var]], axis=1)
plot = data.plot.scatter(x=var, y='timeAvg')
fig = plot.get_figure()
fig.savefig("transitionsPlot")
var = 'Min_tokens_marking'
data = pandas.concat([df['timeAvg'], df[var]], axis=1)
plot = data.plot.scatter(x=var, y='timeAvg')
fig = plot.get_figure()
fig.savefig("minTokensPlot")
var = 'Max_tokens_marking'
data = pandas.concat([df['timeAvg'], df[var]], axis=1)
plot = data.plot.scatter(x=var, y='timeAvg')
fig = plot.get_figure()
fig.savefig("maxTokensPlot")
var = 'Concurrent_Units'
data = pandas.concat([df['timeAvg'], df[var]], axis=1)
plot = data.plot.scatter(x=var, y='timeAvg')
fig = plot.get_figure()
fig.savefig("concurrentUnitsPlot")
var = 'Places'
data = pandas.concat([df['timeAvg'], df[var]], axis=1)
plot = data.plot.scatter(x=var, y='timeAvg')
fig = plot.get_figure()
fig.savefig("placesPlot")
var = 'Arcs'
data = pandas.concat([df['timeAvg'], df[var]], axis=1)
plot = data.plot.scatter(x=var, y='timeAvg')
fig = plot.get_figure()
fig.savefig("arcsPlot")
var = 'Markings'
data = pandas.concat([df['timeAvg'], df[var]], axis=1)
plot = data.plot.scatter(x=var, y='timeAvg')
fig = plot.get_figure()
fig.savefig("markingsPlot")
var = 'Firings'
data = pandas.concat([df['timeAvg'], df[var]], axis=1)
plot = data.plot.scatter(x=var, y='timeAvg')
fig = plot.get_figure()
fig.savefig("firingsPlot")
var = 'Exclusives'
data = pandas.concat([df['timeAvg'], df[var]], axis=1)
plot = data.plot.scatter(x=var, y='timeAvg')
fig = plot.get_figure()
fig.savefig("exclusivesPlot")
var = 'timeAvg'
data = pandas.concat([df['timeAvg'], df[var]], axis=1)
plot = data.plot.scatter(x=var, y='timeAvg')
fig = plot.get_figure()
fig.savefig("timePlot")
