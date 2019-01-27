import pandas
import numpy
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import ensemble
from sklearn import neighbors
from sklearn.model_selection import train_test_split

bfssat = pandas.read_csv("bfssatmcc.csv")
bfssat = bfssat[[ 'filename', 'profile.x', 'span.x', 'Min_tokens_marking', 'timeAvg', 'Deadlock' ]]
bfssatRegressor = linear_model.ElasticNet()

bfsfix = pandas.read_csv("bfsfixmcc.csv")
bfsfix = bfsfix[[ 'filename',  'Arcs', 'Conservative', 'Deadlock', 'profile.x', 'span.x', 'Min_tokens_marking', 'timeAvg' ]]
bfsfixRegressor = pipeline.make_pipeline(preprocessing.StandardScaler(), ensemble.ExtraTreesRegressor(n_estimators=100, criterion='mse'))

bfslike = pandas.read_csv("bfslikemcc.csv")
bfslike = bfslike[[ 'filename',  'Max_tokens_place', 'Min_tokens_marking', 'timeAvg' ]]
bfslikeRegressor = pipeline.make_pipeline(preprocessing.StandardScaler(), ensemble.ExtraTreesRegressor(n_estimators=100, criterion='mae'))

bfsloop = pandas.read_csv("bfsloopmcc.csv")
bfsloop = bfsloop[[ 'filename',  'Dead_transitions', 'span.x', 'Max_tokens_place', 'timeAvg' ]]
bfsloopRegressor = linear_model.Ridge()

bfsnone = pandas.read_csv("bfsnonemcc.csv")
bfsnone = bfsnone[[ 'filename',  'Arcs', 'Deadlock', 'Places', 'Max_tokens_place', 'Min_tokens_marking', 'timeAvg' ]]
bfsnoneRegressor = neighbors.KNeighborsRegressor(n_neighbors=1, weights='distance', p=2)

bfsprevfix = pandas.read_csv("bfsprevfixmcc.csv")
bfsprevfix = bfsprevfix[[ 'filename',  'Conservative', 'Deadlock', 'profile.x', 'span.x', 'Max_tokens_place', 'Min_tokens_marking', 'timeAvg' ]]
bfsprevfixRegressor = pipeline.make_pipeline(preprocessing.StandardScaler(), ensemble.ExtraTreesRegressor(n_estimators=100, criterion='mse'))

bfsprevlike = pandas.read_csv("bfsprevlikemcc.csv")
bfsprevlike = bfsprevlike[[ 'filename',  'Conservative', 'Dead_transitions', 'Places', 'Min_tokens_marking', 'timeAvg' ]]
bfsprevlikeRegressor = pipeline.make_pipeline(preprocessing.StandardScaler(), ensemble.ExtraTreesRegressor(n_estimators=100, criterion='mse'))

bfsprevloop = pandas.read_csv("bfsprevloopmcc.csv")
bfsprevloop = bfsprevloop[[ 'filename',  'Conservative', 'Deadlock', 'Transitions', 'Min_tokens_marking', 'timeAvg' ]]
bfsprevloopRegressor = pipeline.make_pipeline(preprocessing.StandardScaler(), ensemble.ExtraTreesRegressor(n_estimators=100, criterion='mae'))

bfsprevnone = pandas.read_csv("bfsprevnonemcc.csv")
bfsprevnone = bfsprevnone[[ 'filename',  'Transitions', 'Min_tokens_marking', 'timeAvg' ]]
bfsprevnoneRegressor = pipeline.make_pipeline(preprocessing.StandardScaler(), ensemble.ExtraTreesRegressor(n_estimators=100, criterion='mae'))

bfsprevsat = pandas.read_csv("bfsprevsatmcc.csv")
bfsprevsat = bfsprevsat[[ 'filename',  'Dead_transitions', 'Deadlock', 'profile.x', 'span.x', 'Min_tokens_marking', 'timeAvg' ]]
bfsprevsatRegressor = linear_model.ElasticNet()

chainfix = pandas.read_csv("chainfixmcc.csv")
chainfix = chainfix[[ 'filename',  'Arcs', 'Conservative', 'Deadlock', 'profile.x', 'span.x', 'Min_tokens_marking', 'timeAvg' ]]
chainfixRegressor = pipeline.make_pipeline(preprocessing.StandardScaler(), ensemble.ExtraTreesRegressor(n_estimators=100, criterion='mse'))

chainlike = pandas.read_csv("chainlikemcc.csv")
chainlike = chainlike[[ 'filename',  'Conservative', 'Transitions', 'profile.x', 'Max_tokens_place', 'Min_tokens_marking', 'timeAvg' ]]
chainlikeRegressor = pipeline.make_pipeline(preprocessing.StandardScaler(), ensemble.ExtraTreesRegressor(n_estimators=100, criterion='mae'))

chainloop = pandas.read_csv("chainloopmcc.csv")
chainloop = chainloop[[ 'filename',  'Conservative', 'Deadlock', 'Min_tokens_marking', 'timeAvg' ]]
chainloopRegressor = pipeline.make_pipeline(preprocessing.StandardScaler(), ensemble.ExtraTreesRegressor(n_estimators=100, criterion='mae'))

chainnone = pandas.read_csv("chainnonemcc.csv")
chainnone = chainnone[[ 'filename',  'Max_tokens_place', 'timeAvg' ]]
chainnoneRegressor = neighbors.KNeighborsRegressor(weights='distance')

chainsat = pandas.read_csv("chainsatmcc.csv")
chainsat = chainsat[[ 'filename',  'Deadlock', 'profile.x', 'span.x', 'Min_tokens_marking', 'timeAvg' ]]
chainsatRegressor = linear_model.ElasticNet()

chainprevfix = pandas.read_csv("chainprevfixmcc.csv")
chainprevfix = chainprevfix[[ 'filename',  'Arcs', 'Conservative', 'Deadlock', 'profile.x', 'span.x', 'Min_tokens_marking', 'timeAvg' ]]
chainprevfixRegressor = pipeline.make_pipeline(preprocessing.StandardScaler(), ensemble.ExtraTreesRegressor(n_estimators=100, criterion='mse'))

chainprevlike = pandas.read_csv("chainprevlikemcc.csv")
chainprevlike = chainprevlike[[ 'filename',  'Dead_transitions', 'Places', 'profile.x', 'Min_tokens_marking', 'timeAvg' ]]
chainprevlikeRegressor = pipeline.make_pipeline(preprocessing.StandardScaler(), ensemble.ExtraTreesRegressor(n_estimators=100, criterion='mse'))

chainprevloop = pandas.read_csv("chainprevloopmcc.csv")
chainprevloop = chainprevloop[[ 'filename',  'Conservative', 'Deadlock', 'Min_tokens_marking', 'timeAvg' ]]
chainprevloopRegressor = pipeline.make_pipeline(preprocessing.StandardScaler(), ensemble.ExtraTreesRegressor(n_estimators=100, criterion='mae'))

chainprevnone = pandas.read_csv("chainprevnonemcc.csv")
chainprevnone = chainprevnone[[ 'filename',  'Conservative', 'Dead_transitions', 'Max_tokens_place', 'timeAvg' ]]
chainprevnoneRegressor = pipeline.make_pipeline(preprocessing.StandardScaler(), ensemble.RandomForestRegressor(n_estimators=100, criterion='mse'))

chainprevsat = pandas.read_csv("chainprevsatmcc.csv")
chainprevsat = chainprevsat[[ 'filename',  'Deadlock', 'span.x', 'timeAvg' ]]
chainprevsatRegressor = linear_model.ElasticNet()

allDatasets = [ (bfssat,'bfssat'), (bfsfix,'bfsfix'), (bfslike,'bfslike'), (bfsloop,'bfsloop'), (bfsnone,'bfsnone'), (bfsprevfix,'bfsprevfix'), (bfsprevlike,'bfsprevlike'), (bfsprevloop,'bfsprevloop'), (bfsprevnone,'bfsprevnone'), (bfsprevsat,'bfsprevsat'), (chainfix,'chainfix'), (chainlike,'chainlike'), (chainloop,'chainloop'), (chainnone,'chainnone'), (chainsat,'chainsat'), (chainprevfix,'chainprevfix'), (chainprevlike,'chainprevlike'), (chainprevloop,'chainprevloop'), (chainprevnone,'chainprevnone'), (chainprevsat,'chainprevsat') ]
allRegressors = [ (bfssatRegressor,'bfssat'), (bfsfixRegressor,'bfsfix'), (bfslikeRegressor,'bfslike'), (bfsloopRegressor,'bfsloop'), (bfsnoneRegressor,'bfsnone'), (bfsprevfixRegressor,'bfsprevfix'), (bfsprevlikeRegressor,'bfsprevlike'), (bfsprevloopRegressor,'bfsprevloop'), (bfsprevnoneRegressor,'bfsprevnone'), (bfsprevsatRegressor,'bfsprevsat'), (chainfixRegressor,'chainfix'), (chainlikeRegressor,'chainlike'), (chainloopRegressor,'chainloop'), (chainnoneRegressor,'chainnone'), (chainsatRegressor,'chainsat'), (chainprevfixRegressor,'chainprevfix'), (chainprevlikeRegressor,'chainprevlike'), (chainprevloopRegressor,'chainprevloop'), (chainprevnoneRegressor,'chainprevnone'), (chainprevsatRegressor,'chainprevsat') ]

def getBestStrat(filename):
    bestTime = numpy.inf
    bestStrat = None
    for dataset in allDatasets:
        df = dataset[0].loc[dataset[0]['filename'] == filename, 'timeAvg']
        if len(df) > 0 and df.iloc[0] < bestTime:
            bestTime = df.iloc[0]
            bestStrat = dataset[1]
    return bestStrat

def getSmallestDF():
    size = numpy.inf
    smallestDF = None
    for dataset in allDatasets:
        if size > len(dataset[0].index):
            size = len(dataset[0].index)
            smallestDF = dataset[0]
    return smallestDF

def toName(dataset):
    for t in allDatasets:
        if dataset.equals(t[0]):
            return t[1]

def allToSmallest(filenames):
    newDataset = []
    for t in allDatasets:
        t = (t[0][t[0].filename.isin(filenames)], t[1])
        newDataset.append(t)
    return newDataset

def getStratDf(name):
    return [item for item in allDatasets if item[1] == name][0][0]

def getAverageCompletionTime(df):
    count = 0
    totalTime = 0
    timeFrame = df.timeAvg
    for time in timeFrame:
        print(time)
        count += 1
        totalTime = time
    return totalTime / count

timeTuple = {}
for t in allDatasets:
    time = getAverageCompletionTime(t[0])
    timeTuple[t[1]] = time

print(timeTuple)

smallest = getSmallestDF()
#count = {}
#for index, row in smallest.iterrows():
#    filename = row['filename']
#    bestStrat = getBestStrat(filename)
#    if bestStrat not in count:
#        count[bestStrat] = 1
#    else:
#        count[bestStrat] = count[bestStrat] + 1
#cMax = 0
minTime = numpy.inf
bestSet = None
for t in timeTuple:
    if timeTuple[t] < minTime:
        minTime = timeTuple[t]
        bestSet = t
print("Best strat: " + str(bestSet))
filenames = smallest['filename'].tolist()
allDatasets = allToSmallest(filenames)
smallest = getSmallestDF()
filenames = smallest['filename'].tolist()
allDatasets = allToSmallest(filenames)
smallest = getSmallestDF()
filenames = smallest['filename'].tolist()
allDatasets = allToSmallest(filenames)
print(smallest)

right = 0
wrong = 0
for i in range(0, 100):
    train, test = train_test_split(smallest['filename'], test_size=0.33)

    predTimesDict = {}
    regressorScores = []
    for dfTuple in allDatasets:
        df = dfTuple[0]
        name = dfTuple[1]
        regressor = [item for item in allRegressors if item[1] == name][0]
        regressor = regressor[0]
        trainData = df[df.filename.isin(train)]
        testData = df[df.filename.isin(test)]
        X_train = trainData.drop(['filename','timeAvg'], axis=1)
        y_train = trainData.timeAvg
        X_test = testData.drop(['filename','timeAvg'], axis=1)
        y_test = testData.timeAvg
        regressor.fit(X_train, y_train)
        regressorScores.append((name, regressor.score(X_test, y_test)))
        predTimes = regressor.predict(X_test)
        testNames = testData.filename.tolist()
        filenamePredTimes = []
        for i in range(0,len(testNames)):
            modelName = testNames[i]
            predTime = predTimes[i]
            t = (modelName, predTime)
            filenamePredTimes.append(t)
        predTimesDict[name] = filenamePredTimes

    bestStrat = {}
    for stratName in predTimesDict:
        predTimes = predTimesDict[stratName]
        for t in predTimes:
            modelName = t[0]
            predTime = t[1]
            if modelName not in bestStrat or predTime < bestStrat[modelName][1]:
                bestStrat[modelName] = (stratName, t[1])

    bestDf = [item for item in allDatasets if item[1] == bestSet][0][0]
    #print(regressorScores)
    wrongCount = 0
    rightCount = 0
    for modelName in bestStrat:
        stratDf = getStratDf(bestStrat[modelName][0])
        bestPredStratName = bestStrat[modelName][0]
        bestPredStratTime = stratDf.loc[stratDf['filename'] == modelName].iloc[0].timeAvg
        bestOverallStratTime = bestDf.loc[bestDf['filename'] == modelName].iloc[0].timeAvg
        if bestPredStratTime <= bestOverallStratTime:
            rightCount = rightCount + 1
        else:
            wrongCount = wrongCount + 1
        print("Model name: " + str(modelName) + " best strat: " + str(bestPredStratName) + " time: " + str(bestPredStratTime) + " time in overall strat: " + str(bestOverallStratTime))
    right += rightCount
    wrong += wrongCount
    print("Right: " + str(rightCount) + " Wrong: " + str(wrongCount))
right = right/100
wrong = wrong/100

print("Average Right: " + str(right) + " Average wrong: " + str(wrong))

