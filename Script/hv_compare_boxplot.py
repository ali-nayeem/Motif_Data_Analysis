import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#from pygmo import hypervolume
from pymoo.factory import get_performance_indicator

reference_points = np.ones((1, 2)) * 1.2
hv = get_performance_indicator("hv", ref_point=reference_points[0])

root = '../Data/'
outDir = '../Output/HVBoxplot/'
columns = [ 'Method', 'Dataset', 'Motif Length','Similarity', 'Entropy']  # ['Entropy', 'TC', 'Gap', 'SimG', 'SimNG', 'GapCon']
algoList = [ 'NSGAII', 'NSGAII-PM', 'NSGAII-PC', 'NSGAII-PMC']
algoDirDic = {'ABC':'ABC', 'NSGAII':'NSGA-DefaultMutation-DefaultCrossover', 'NSGAII-PC':'NSGA-DefaultMutation-MyCrossover', 'NSGAII-PM':'NSGA-MyMutation-DefaultCrossover', 'NSGAII-PMC':'NSGA-MyMutation-MyCrossover'} #NSGA-II',
dataList = [ "hm03","hm09g","yst04r","yst08r","dm01g","dm03g"]
motifLenList = [16,18,20,22]
repCount = 10
runCount = 20
genList = list(range(0, 1000, 40))
ref_point = [1.2, 1.2]
popSize = 200


for data in dataList:
    gdf = pd.DataFrame(columns=['Method', 'Motif Length', 'Hypervolume'])
    for motLen in motifLenList:
         #df_algo['gen'] = genList
         for algo in algoList:
             df_run = pd.DataFrame()
             #avg_hv = 0
             for runId in range(0, runCount):
                pf = np.loadtxt(root + algoDirDic[algo] + '/Pareto front/' + data + '/' + str(motLen) + '/' + 'run' + str(runId) +'.txt')
                #print(root + algoDirDic[algo] + '/Pareto front/' + data + '/' + str(motLen) + '/' + 'run' + str(runId) +'.txt')
                pf = np.multiply(pf, -1)
                # print( str(runId) + ' ' + algo + ' ' +  data + ' ' + str(motLen))
                # if runId == 18:
                #     print(pf)
                #!!!! TMP FIX !!!!!!
                # if len(pf) < popSize:
                #     continue
                #hv = hypervolume(pf)
                #avg_hv =  avg_hv + hv.compute(ref_point)
                row = []
                row.append(algo)
                row.append(motLen)
                row.append(hv.calc(pf)) #hv.calc(F)  hv.compute(ref_point)
                gdf.loc[len(gdf)] = row
    sns.set(style="whitegrid")
    sns.set_context("talk")

    g = sns.boxplot(x="Motif Length", y="Hypervolume",
                     hue="Method", data=gdf, linewidth=1.5)  # , linewidth=1
    g.legend_.remove()
    plt.legend(loc='upper center', bbox_to_anchor=(0.42, 1.09),
                ncol=4, fancybox=True, prop={'size': 11})  # prop={'size': 8}, , shadow=True   fontsize='small'
    # plt.show()
    plt.savefig(outDir + data + '_hv_all_boxplot.png', format='png', bbox_inches='tight')
    plt.clf()

     # plt.xlabel('Generation')
     # plt.ylabel('Hypervolume')
     # plt.legend()
     # # plt.show()
     # # plt.clf()
     # plt.savefig(outDir + data + '_' + str(motLen) +'_hv_progress.png', format='png', bbox_inches='tight')
     # plt.clf()
