import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from pygmo import hypervolume

root = '../Data/'
outDir = 'output/'
columns = [ 'Method', 'Dataset', 'Motif Length','Similarity', 'Entropy']  # ['Entropy', 'TC', 'Gap', 'SimG', 'SimNG', 'GapCon']
algoList = [ 'NSGAII', 'NSGAII-PM', 'NSGA-II-PC', 'NSGAII-PMC']
algoDirDic = {'ABC':'ABC', 'NSGAII':'NSGA-DefaultMutation-DefaultCrossover', 'NSGA-II-PC':'NSGA-DefaultMutation-MyCrossover', 'NSGAII-PM':'NSGA-MyMutation-DefaultCrossover', 'NSGAII-PMC':'NSGA-MyMutation-MyCrossover'} #NSGA-II',
toolList = ['ABC']
dataList = ["hm03","hm09g","yst04r","yst08r","dm01g","dm03g"]
motifLenList = [16,18,20,22]
repCount = 10
runCount = 20
simCol = 0
entrpCol = 1

for data in dataList:
    gdf = pd.DataFrame(columns=columns)
    for algo in algoList:
        for motLen in motifLenList:
            ldf = pd.read_csv(root + algoDirDic[algo] + '/Best Outputs/' + data + '/' + str(motLen) + '.txt', delimiter=r"\s+", header=None,
                        names=['Similarity', 'Entropy'], index_col=False)
            ldf['Method'] = algo
            ldf['Dataset'] = data
            ldf['Motif Length'] = motLen
            gdf = gdf.append(ldf, ignore_index=True, sort=True)
    for tool in toolList:



