import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

root = '../Data/'
outDir = '../Output/'
columns = [  'NSGAII', 'NSGAII-PM', 'NSGAII-PC', 'NSGAII-PMC', 'ABC']  # ['Entropy', 'TC', 'Gap', 'SimG', 'SimNG', 'GapCon']
algoList = [ 'NSGAII', 'NSGAII-PM', 'NSGAII-PC', 'NSGAII-PMC']
algoDirDic = {'ABC':'ABC', 'NSGAII':'NSGA-DefaultMutation-DefaultCrossover', 'NSGAII-PC':'NSGA-DefaultMutation-MyCrossover', 'NSGAII-PM':'NSGA-MyMutation-DefaultCrossover', 'NSGAII-PMC':'NSGA-MyMutation-MyCrossover'} #NSGA-II',
toolList = ['ABC']
dataList = ["hm03","hm09g","yst04r","yst08r","dm01g","dm03g"]
motifLenList = [16,18,20,22]
repCount = 10
runCount = 20
simCol = 0
entrpCol = 1
toolFitnessList = ['Similarity', 'Entropy']
fitnessId = 1
indexList = list(range(0, len(motifLenList) * len(dataList)))
toolColList = [['Motif Length', 'Similarity'], ['Entropy']]
gdf = pd.DataFrame(index=indexList, columns=columns)
for algo in algoList:
    objList = []
    for data in dataList:
        for motLen in motifLenList:
            obj = np.genfromtxt(root + algoDirDic[algo] + '/Best Outputs/' + data + '/' + str(motLen) + '.txt', usecols=fitnessId)
            objList.append(np.mean(obj))
        #ldf = pd.read_csv(columns=columns)
    gdf[algo] = objList

# for fitness in toolFitnessList:
for tool in toolList:
    objList = []
    for data in dataList:
       for motLen in motifLenList:
        ldf = pd.read_csv(root + tool + '/Best Outputs/' + toolFitnessList[fitnessId] + '/' + data + '.txt.txt', delimiter=r"\s+", header=None,
                    names=toolColList[fitnessId], index_col=False)
        objList.append(ldf[toolFitnessList[fitnessId]].mean())

        # entropy = np.genfromtxt(root + tool + '/Best Outputs/' + 'Entropy' + '/' + data + '.txt.txt',
        #                    delimiter=r"\s+")
        # ldf['Entropy'] = entropy
        # gdf = gdf.append(ldf, ignore_index=True, sort=True)
    gdf[tool] = objList
    #plt.show()
gdf.to_csv(outDir+'stat_test_'+toolFitnessList[fitnessId]+'.csv')