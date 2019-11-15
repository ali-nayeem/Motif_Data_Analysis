import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

root = '../Data/'
outDir = '../Output/HV/'
columns = [ 'Method', 'Dataset', 'Motif Length','Similarity', 'Entropy']  # ['Entropy', 'TC', 'Gap', 'SimG', 'SimNG', 'GapCon']
algoList = [ 'NSGAII', 'NSGAII-PM', 'NSGAII-PC', 'NSGAII-PMC']
algoDirDic = {'ABC':'ABC', 'NSGAII':'NSGA-DefaultMutation-DefaultCrossover', 'NSGAII-PC':'NSGA-DefaultMutation-MyCrossover', 'NSGAII-PM':'NSGA-MyMutation-DefaultCrossover', 'NSGAII-PMC':'NSGA-MyMutation-MyCrossover'} #NSGA-II',
dataList = ["hm03","hm09g","yst04r","yst08r","dm01g","dm03g"]
motifLenList = [16,18,20,22]
repCount = 10
runCount = 20
genList = list(range(0, 1000, 40))
sns.set(style="whitegrid")
sns.set_context("talk")
for data in dataList:
    gdf = pd.DataFrame(columns=columns)
    for motLen in motifLenList:
         df_algo = pd.DataFrame()
         #df_algo['gen'] = genList
         for algo in algoList:
             df_run = pd.DataFrame()
             for runId in range(0, runCount):
                hv = np.genfromtxt(root + algoDirDic[algo] + '/Hv-data/' + data + '/' + str(motLen) + '/' + 'run' + str(runId) +'.txt', delimiter=r"\s+")
                if len(hv) > len(genList):
                    hv = hv[-len(genList):]
                df_run['run' + str(runId)] = hv
             df_algo[algo] = df_run.mean(1)
             plt.plot(genList, df_algo[algo], label=algo)
    # plt.figure(figsize=(20,3))
    # plt.plot(genList, mat[0, :], label=algoLabelDic[algoList[0]])
    # plt.plot(genList, mat[1, :], label=algoLabelDic[algoList[1]])
         #plt.yscale('log')
         plt.xlabel('Generation')
         plt.ylabel('Hypervolume')
         plt.legend()
         # plt.show()
         # plt.clf()
         plt.savefig(outDir + data + '_' + str(motLen) +'_hv_progress.png', format='png', bbox_inches='tight')
         plt.clf()
