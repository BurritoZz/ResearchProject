import pandas
import numpy
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import pipeline
from sklearn import ensemble
import matplotlib.pyplot as plt
import itertools
data = pandas.read_csv("bfslikemcc.csv")
cols_of_interest = [ 'timeAvg', 'profile.x', 'span.x', 'Places', 'Transitions', 'Arcs', 'Ordinary', 'Simple_free_choice', 'Extended_free_choice', 'State_machine', 'Marked_graph', 'Connected', 'Strongly_connected', 'Source_place', 'Sink_place', 'Source_transitions', 'Sink_transitions', 'Loop_free', 'Conservative', 'Subconservative', 'Nested_units', 'Safe', 'Deadlock', 'Quasi_live', 'Live', 'Markings', 'Firings', 'Max_tokens_place', 'Min_tokens_marking', 'Max_tokens_marking', 'Dead_transitions', 'Concurrent_Units', 'Exclusives' ]
data = data[cols_of_interest]
data = data.replace([numpy.inf, -numpy.inf], numpy.nan)
data = data.dropna()
#feature_cols = [ 'profile', 'span', 'Places', 'Transitions', 'Arcs', 'Ordinary', 'Simple_free_choice', 'Extended_free_choice', 'State_machine', 'Marked_graph', 'Connected', 'Strongly_connected', 'Source_place', 'Sink_place', 'Source_transitions', 'Sink_transitions', 'Loop_free', 'Conservative', 'Subconservative', 'Nested_units', 'Safe', 'Deadlock', 'Quasi_live', 'Live', 'Markings', 'Firings', 'Max_tokens_place', 'Min_tokens_marking', 'Max_tokens_marking', 'Dead_transitions', 'Concurrent_Units', 'Exclusives' ]
#feature_cols = [ 'profile', 'span', 'Transitions', 'Min_tokens_marking', 'Concurrent_Units' ]
feature_cols = [ 'Arcs', 'Conservative', 'Dead_transitions', 'Deadlock', 'Places', 'Transitions', 'profile.x', 'span.x', 'Max_tokens_place', 'Min_tokens_marking' ]

#X = data.loc[:, feature_cols]
#y = data.timeAvg
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#data.to_csv("test.csv")
#print("Decision tree regressor")
#print("mse")
#reg = pipeline.make_pipeline(preprocessing.StandardScaler(), DecisionTreeRegressor(criterion='mse', splitter='random'))
#reg.fit(X_train, y_train)
#print(reg.score(X_test, y_test))
#x = data['span']
#y = data['timeAvg']
#fig, ax = plt.subplots()
#ax.scatter(x, y)
#ax.set_xlabel('span')
#ax.set_ylabel('timeAvg')
#plt.xlim(-10, 10000)
#plt.savefig("span")

x = data.timeAvg
plt.hist(x, normed=True, bins=100)
plt.ylabel("Number of occurrences")
plt.xlabel("Average completion time")
plt.savefig("Hist")

#for variable in feature_cols:
#    x = data[variable]
#    y = data['timeAvg']
#    fig, ax = plt.subplots()
#    ax.scatter(x, y)
#    ax.set_xlabel(variable)
#    ax.set_ylabel('timeAvg')
#    plt.savefig(variable + ".png")



#print(data.isnull().values.any())

#files_to_analyse = [ 'bfssatmcc.csv', 'bfsfixmcc.csv', 'bfslikemcc.csv', 'bfsloopmcc.csv', 'bfsnonemcc.csv', 'bfsprevfixmcc.csv', 'bfsprevlikemcc.csv', 'bfsprevloopmcc.csv', 'bfsprevnonemcc.csv', 'bfsprevsatmcc.csv', 'chainfixmcc.csv', 'chainlikemcc.csv', 'chainloopmcc.csv', 'chainnonemcc.csv', 'chainsatmcc.csv', 'chainprevfixmcc.csv', 'chainprevlikemcc.csv', 'chainprevloopmcc.csv', 'chainprevnonemcc.csv', 'chainprevsatmcc.csv' ]
#
#def powerset(iterable):
#    s = list(iterable)
#    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))
#
##endResult = ["Filename: bfssatmcc.csv Max score: 0.4648942719404152 Best subset: ('Deadlock', 'profile', 'span', 'Min_tokens_marking') Best alg: ElasticNet", "Filename: bfsfixmcc.csv Max score: 0.491818385759499 Best subset: ('Arcs', 'Conservative', 'Deadlock', 'profile', 'span', 'Min_tokens_marking') Best alg: ExtraTreesRegressorMSE"]
#endResult = []
#for f in files_to_analyse:
#    data = pandas.read_csv(f)
#    cols_of_interest = [ 'timeAvg', 'profile.x', 'span.x', 'Places', 'Transitions', 'Arcs', 'Ordinary', 'Simple_free_choice', 'Extended_free_choice', 'State_machine', 'Marked_graph', 'Connected', 'Strongly_connected', 'Source_place', 'Sink_place', 'Source_transitions', 'Sink_transitions', 'Loop_free', 'Conservative', 'Subconservative', 'Nested_units', 'Safe', 'Deadlock', 'Quasi_live', 'Live', 'Markings', 'Firings', 'Max_tokens_place', 'Min_tokens_marking', 'Max_tokens_marking', 'Dead_transitions', 'Concurrent_Units', 'Exclusives' ]
#    data = data[cols_of_interest]
#    data = data.replace([numpy.inf, -numpy.inf], numpy.nan)
#    data = data.dropna()
#    maxScore = -100
#    bestSubset = ()
#    bestAlg = ""
#    pSet = powerset(feature_cols)
#    for subset in pSet:
#        if subset == ():
#            continue
#        print(f)
#        print(subset)
#        X = data.loc[:, subset]
#        y = data.timeAvg
#        score = 0
#        for i in range(0, 10):
#            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#            reg = linear_model.Ridge()
#            reg.fit(X_train, y_train)
#            score += reg.score(X_test, y_test)
#        score = score / 10
#        if score > maxScore:
#            maxScore = score
#            bestSubset = subset
#            bestAlg = "Ridge"
#        score = 0
#        for i in range(0, 10):
#            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#            reg = linear_model.Lasso()
#            reg.fit(X_train, y_train)
#            score += reg.score(X_test, y_test)
#        score = score / 10
#        if score > maxScore:
#            maxScore = score
#            bestSubset = subset
#            bestAlg = "Lasso"
#        score = 0
#        for i in range(0, 10):
#            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#            reg = linear_model.ElasticNet()
#            reg.fit(X_train, y_train)
#            score += reg.score(X_test, y_test)
#        score = score / 10
#        if score > maxScore:
#            maxScore = score
#            bestSubset = subset
#            bestAlg = "ElasticNet"
#        score = 0
#        for i in range(0, 10):
#            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#            reg = linear_model.BayesianRidge()
#            reg.fit(X_train, y_train)
#            score += reg.score(X_test, y_test)
#        score = score / 10
#        if score > maxScore:
#            maxScore = score
#            bestSubset = subset
#            bestAlg = "BayesianRidge"
#        score = 0
#        for i in range(0, 10):
#            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#            reg = pipeline.make_pipeline(preprocessing.StandardScaler(), linear_model.ARDRegression())
#            reg.fit(X_train, y_train)
#            score += reg.score(X_test, y_test)
#        score = score / 10
#        if score > maxScore:
#            maxScore = score
#            bestSubset = subset
#            bestAlg = "ARDRegression"
#        score = 0
#        for i in range(0, 10):
#            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#            reg = svm.SVR(kernel='rbf')
#            reg.fit(X_train, y_train)
#            score += reg.score(X_test, y_test)
#        score = score / 10
#        if score > maxScore:
#            maxScore = score
#            bestSubset = subset
#            bestAlg = "SVR"
#        score = 0
#        for i in range(0, 10):
#            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#            reg = KNeighborsRegressor(n_neighbors=1, weights='distance', p=2)
#            reg.fit(X_train, y_train)
#            score += reg.score(X_test, y_test)
#        score = score / 10
#        if score > maxScore:
#            maxScore = score
#            bestSubset = subset
#            bestAlg = "KNeighbors One neighbour"
#        score = 0
#        for i in range(0, 10):
#            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#            reg = KNeighborsRegressor(weights='distance')
#            reg.fit(X_train, y_train)
#            score += reg.score(X_test, y_test)
#        score = score / 10
#        if score > maxScore:
#            maxScore = score
#            bestSubset = subset
#            bestAlg = "Kneighbours default"
#        score = 0
#        for i in range(0, 10):
#            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#            reg = pipeline.make_pipeline(preprocessing.StandardScaler(), DecisionTreeRegressor(criterion='mse', splitter='random'))
#            reg.fit(X_train, y_train)
#            score += reg.score(X_test, y_test)
#        score = score / 10
#        if score > maxScore:
#            maxScore = score
#            bestSubset = subset
#            bestAlg = "DecisionTreeMSE"
#        score = 0
#        for i in range(0, 10):
#            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#            reg = pipeline.make_pipeline(preprocessing.StandardScaler(), DecisionTreeRegressor(criterion='friedman_mse', splitter='random'))
#            reg.fit(X_train, y_train)
#            score += reg.score(X_test, y_test)
#        score = score / 10
#        if score > maxScore:
#            maxScore = score
#            bestSubset = subset
#            bestAlg = "DecisionTreeFriedman"
#        score = 0
#        for i in range (0, 10):
#            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#            reg = pipeline.make_pipeline(preprocessing.StandardScaler(), DecisionTreeRegressor(criterion='mae', splitter='random'))
#            reg.fit(X_train, y_train)
#            score += reg.score(X_test, y_test)
#        score = score / 10
#        if score > maxScore:
#            maxScore = score
#            bestSubset = subset
#            bestAlg = "DecisionTreeMAE"
#        score = 0
#        for i in range (0, 10):
#            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#            reg = ensemble.AdaBoostRegressor(n_estimators=100)
#            reg.fit(X_train, y_train)
#            score += reg.score(X_test, y_test)
#        score = score / 10
#        if score > maxScore:
#            maxScore = score
#            bestSubset = subset
#            bestAlg = "AdaBoostRegressor"
#        score = 0
#        for i in range (0, 10):
#            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#            reg = ensemble.BaggingRegressor(n_estimators=100)
#            reg.fit(X_train, y_train)
#            score += reg.score(X_test, y_test)
#        score = score / 10
#        if score > maxScore:
#            maxScore = score
#            bestSubset = subset
#            bestAlg = "BaggingRegressor"
#        score = 0
#        for i in range (0, 10):
#            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#            reg = pipeline.make_pipeline(preprocessing.StandardScaler(), ensemble.ExtraTreesRegressor(n_estimators=100, criterion='mse'))
#            reg.fit(X_train, y_train)
#            score += reg.score(X_test, y_test)
#        score = score / 10
#        if score > maxScore:
#            maxScore = score
#            bestSubset = subset
#            bestAlg = "ExtraTreesRegressorMSE"
#        score = 0
#        for i in range (0, 10):
#            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#            reg = pipeline.make_pipeline(preprocessing.StandardScaler(), ensemble.ExtraTreesRegressor(n_estimators=100, criterion='mae'))
#            reg.fit(X_train, y_train)
#            score += reg.score(X_test, y_test)
#        score = score / 10
#        if score > maxScore:
#            maxScore = score
#            bestSubset = subset
#            bestAlg = "ExtraTreesRegressorMAE"
#        score = 0
#        for i in range (0, 10):
#            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#            reg = ensemble.GradientBoostingRegressor()
#            reg.fit(X_train, y_train)
#            score += reg.score(X_test, y_test)
#        score = score / 10
#        if score > maxScore:
#            maxScore = score
#            bestSubset = subset
#            bestAlg = "GradientBoostingRegressor"
#        score = 0
#        for i in range (0, 10):
#            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#            reg = pipeline.make_pipeline(preprocessing.StandardScaler(), ensemble.RandomForestRegressor(n_estimators=100, criterion='mse'))
#            reg.fit(X_train, y_train)
#            score += reg.score(X_test, y_test)
#        score = score / 10
#        if score > maxScore:
#            maxScore = score
#            bestSubset = subset
#            bestAlg = "RandomForestMSE"
#        score = 0
#        for i in range (0, 10):
#            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#            reg = pipeline.make_pipeline(preprocessing.StandardScaler(), ensemble.RandomForestRegressor(n_estimators=100, criterion='mae'))
#            reg.fit(X_train, y_train)
#            score += reg.score(X_test, y_test)
#        score = score / 10
#        if score > maxScore:
#            maxScore = score
#            bestSubset = subset
#            bestAlg = "RandomForestMAE"
#        print("Max score: " + str(maxScore))
#        print("Best subset: " + str(bestSubset))
#        print("Best alg: " + bestAlg)
#    endResult = endResult + ["Filename: " + f + " Max score: " + str(maxScore) + " Best subset: " + str(bestSubset) + " Best alg: " + bestAlg]
#    print(endResult)
#print(endResult)

#print(subs(feature_cols))
#
#print(data)
#
#print("Ridge")
#reg = linear_model.Ridge()
#reg.fit(X_train, y_train)
#print(reg.score(X_test, y_test))
#
#print("Lasso")
#reg = linear_model.Lasso()
#reg.fit(X_train, y_train)
#print(reg.score(X_test, y_test))
#
#print("ElasticNet")
#reg = linear_model.ElasticNet()
#reg.fit(X_train, y_train)
#print(reg.score(X_test, y_test))
#
#
#print("BayesianRidge")
#reg = linear_model.BayesianRidge()
#reg.fit(X_train, y_train)
#print(reg.score(X_test, y_test))
#
#print("ARDRegression")
#reg = pipeline.make_pipeline(preprocessing.StandardScaler(), linear_model.ARDRegression())
#reg.fit(X_train, y_train)
#print(reg.score(X_test, y_test))
#
#print("SVR")
#reg = svm.SVR(kernel='rbf')
#reg.fit(X_train, y_train)
#print(reg.score(X_test, y_test))
#
#print("KNeighbors")
#reg = KNeighborsRegressor(n_neighbors=1, weights='distance', p=2)
#reg.fit(X_train, y_train)
#print(reg.score(X_test, y_test))
#
#
#print("friedman")
#reg = pipeline.make_pipeline(preprocessing.StandardScaler(), DecisionTreeRegressor(criterion='friedman_mse', splitter='random'))
#reg.fit(X_train, y_train)
#print(reg.score(X_test, y_test))
#
#print("mae")
#reg = pipeline.make_pipeline(preprocessing.StandardScaler(), DecisionTreeRegressor(criterion='mae', splitter='random'))
#reg.fit(X_train, y_train)
#print(reg.score(X_test, y_test))

