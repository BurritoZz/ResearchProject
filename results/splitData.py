import pandas

data = pandas.read_csv("sat+fix.csv")
bfs = data[data.order == 'bfs']
bfsprev = data[data.order == 'bfs-prev']
chain = data[data.order == 'chain']
chainprev = data[data.order == 'chain-prev']

#bfslike = bfs[bfs.saturation == 'sat-like']
#bfsloop = bfs[bfs.saturation == 'sat-loop']
#bfsnone = bfs[bfs.saturation == 'none']
bfssat = bfs[bfs.saturation == 'sat']
bfsfix = bfs[bfs.saturation == 'sat-fix']
#bfsprevlike = bfsprev[bfsprev.saturation == 'sat-like']
#bfsprevloop = bfsprev[bfsprev.saturation == 'sat-loop']
#bfsprevnone = bfsprev[bfsprev.saturation == 'none']
bfsprevsat = bfsprev[bfsprev.saturation == 'sat']
bfsprevfix = bfsprev[bfsprev.saturation == 'sat-fix']
#chainlike = chain[chain.saturation == 'sat-like']
#chainloop = chain[chain.saturation == 'sat-loop']
#chainnone = chain[chain.saturation == 'none']
chainsat = chain[chain.saturation == 'sat']
chainfix = chain[chain.saturation == 'sat-fix']
#chainprevlike = chainprev[chainprev.saturation == 'sat-like']
#chainprevloop = chainprev[chainprev.saturation == 'sat-loop']
#chainprevnone = chainprev[chainprev.saturation == 'none']
chainprevsat = chainprev[chainprev.saturation == 'sat']
chainprevfix = chainprev[chainprev.saturation == 'sat-fix']


#bfslike.to_csv("bfslike.csv", index=False)
#bfsloop.to_csv("bfsloop.csv", index=False)
#bfsnone.to_csv("bfsnone.csv", index=False)
bfssat.to_csv("bfssat.csv", index=False)
bfsfix.to_csv("bfsfix.csv", index=False)
#bfsprevlike.to_csv("bfsprevlike.csv", index=False)
#bfsprevloop.to_csv("bfsprevloop.csv", index=False)
#bfsprevnone.to_csv("bfsprevnone.csv", index=False)
bfsprevsat.to_csv("bfsprevsat.csv", index=False)
bfsprevfix.to_csv("bfsprevfix.csv", index=False)
#chainlike.to_csv("chainlike.csv", index=False)
#chainloop.to_csv("chainloop.csv", index=False)
#chainnone.to_csv("chainnone.csv", index=False)
chainsat.to_csv("chainsat.csv", index=False)
chainfix.to_csv("chainfix.csv", index=False)
#chainprevlike.to_csv("chainprevlike.csv", index=False)
#chainprevloop.to_csv("chainprevloop.csv", index=False)
#chainprevnone.to_csv("chainprevnone.csv", index=False)
chainprevsat.to_csv("chainprevsat.csv", index=False)
chainprevfix.to_csv("chainprevfix.csv", index=False)

