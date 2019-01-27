import pandas
import math

mcc = pandas.read_csv("mccNew.csv")
mcc['filename']=mcc['filename'].astype(str)
cols = [ 'Max_tokens_marking', 'Dead_transitions', 'Concurrent_Units', 'Exclusives' ]
idx = mcc.index[mcc['Exclusives'].isnull()]
mcc.loc[idx, cols] = mcc.loc[idx, cols].shift(1, axis=1)
mcc.loc[idx, ['Min_tokens_marking','Max_tokens_marking']] = mcc['Min_tokens_marking']

data = pandas.read_csv("bfslikemerged.csv")
data['filename']=data['filename'].astype(str)
data = pandas.merge(data, mcc, how='inner', on='filename')
data.to_csv("bfslikemcc.csv", index=False)

data = pandas.read_csv("bfsloopmerged.csv")
data['filename']=data['filename'].astype(str)
data = pandas.merge(data, mcc, how='inner', on='filename')
data.to_csv("bfsloopmcc.csv", index=False)

data = pandas.read_csv("bfsnonemerged.csv")
data['filename']=data['filename'].astype(str)
data = pandas.merge(data, mcc, how='inner', on='filename')
data.to_csv("bfsnonemcc.csv", index=False)

data = pandas.read_csv("bfssatmerged.csv")
data['filename']=data['filename'].astype(str)
data = pandas.merge(data, mcc, how='inner', on='filename')
data.to_csv("bfssatmcc.csv", index=False)

data = pandas.read_csv("bfsfixmerged.csv")
data['filename']=data['filename'].astype(str)
data = pandas.merge(data, mcc, how='inner', on='filename')
data.to_csv("bfsfixmcc.csv", index=False)

data = pandas.read_csv("bfsprevlikemerged.csv")
data['filename']=data['filename'].astype(str)
data = pandas.merge(data, mcc, how='inner', on='filename')
data.to_csv("bfsprevlikemcc.csv", index=False)

data = pandas.read_csv("bfsprevloopmerged.csv")
data['filename']=data['filename'].astype(str)
data = pandas.merge(data, mcc, how='inner', on='filename')
data.to_csv("bfsprevloopmcc.csv", index=False)

data = pandas.read_csv("bfsprevnonemerged.csv")
data['filename']=data['filename'].astype(str)
data = pandas.merge(data, mcc, how='inner', on='filename')
data.to_csv("bfsprevnonemcc.csv", index=False)

data = pandas.read_csv("bfsprevsatmerged.csv")
data['filename']=data['filename'].astype(str)
data = pandas.merge(data, mcc, how='inner', on='filename')
data.to_csv("bfsprevsatmcc.csv", index=False)

data = pandas.read_csv("bfsprevfixmerged.csv")
data['filename']=data['filename'].astype(str)
data = pandas.merge(data, mcc, how='inner', on='filename')
data.to_csv("bfsprevfixmcc.csv", index=False)

data = pandas.read_csv("chainlikemerged.csv")
data['filename']=data['filename'].astype(str)
data = pandas.merge(data, mcc, how='inner', on='filename')
data.to_csv("chainlikemcc.csv", index=False)

data = pandas.read_csv("chainloopmerged.csv")
data['filename']=data['filename'].astype(str)
data = pandas.merge(data, mcc, how='inner', on='filename')
data.to_csv("chainloopmcc.csv", index=False)

data = pandas.read_csv("chainnonemerged.csv")
data['filename']=data['filename'].astype(str)
data = pandas.merge(data, mcc, how='inner', on='filename')
data.to_csv("chainnonemcc.csv", index=False)

data = pandas.read_csv("chainsatmerged.csv")
data['filename']=data['filename'].astype(str)
data = pandas.merge(data, mcc, how='inner', on='filename')
data.to_csv("chainsatmcc.csv", index=False)

data = pandas.read_csv("chainfixmerged.csv")
data['filename']=data['filename'].astype(str)
data = pandas.merge(data, mcc, how='inner', on='filename')
data.to_csv("chainfixmcc.csv", index=False)

data = pandas.read_csv("chainprevlikemerged.csv")
data['filename']=data['filename'].astype(str)
data = pandas.merge(data, mcc, how='inner', on='filename')
data.to_csv("chainprevlikemcc.csv", index=False)

data = pandas.read_csv("chainprevloopmerged.csv")
data['filename']=data['filename'].astype(str)
data = pandas.merge(data, mcc, how='inner', on='filename')
data.to_csv("chainprevloopmcc.csv", index=False)

data = pandas.read_csv("chainprevnonemerged.csv")
data['filename']=data['filename'].astype(str)
data = pandas.merge(data, mcc, how='inner', on='filename')
data.to_csv("chainprevnonemcc.csv", index=False)

data = pandas.read_csv("chainprevsatmerged.csv")
data['filename']=data['filename'].astype(str)
data = pandas.merge(data, mcc, how='inner', on='filename')
data.to_csv("chainprevsatmcc.csv", index=False)

data = pandas.read_csv("chainprevfixmerged.csv")
data['filename']=data['filename'].astype(str)
data = pandas.merge(data, mcc, how='inner', on='filename')
data.to_csv("chainprevfixmcc.csv", index=False)
