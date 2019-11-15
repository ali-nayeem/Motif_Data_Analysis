import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#from pygmo import hypervolume
from pymoo.factory import get_performance_indicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

reference_points = np.ones((1, 2)) * 1.2
hv = get_performance_indicator("hv", ref_point=reference_points[0])

root = '../Data/'
outDir = '../Output/HVBoxplot/'
columns = [ 'Method', 'Dataset', 'Motif Length','Similarity', 'Entropy']  # ['Entropy', 'TC', 'Gap', 'SimG', 'SimNG', 'GapCon']
algoList = [ 'NSGAII', 'NSGAII-PM', 'NSGAII-PC', 'NSGAII-PMC']
algoDirDic = {'ABC':'ABC', 'NSGAII':'NSGA-DefaultMutation-DefaultCrossover', 'NSGAII-PC':'NSGA-DefaultMutation-MyCrossover', 'NSGAII-PM':'NSGA-MyMutation-DefaultCrossover', 'NSGAII-PMC':'NSGA-MyMutation-MyCrossover'} #NSGA-II',
markerDic = {'ABC1':'+', 'ABC1':'x','NSGAII':"s", 'NSGAII-PC':'*', 'NSGAII-PM':'o', 'NSGAII-PMC':'^'} #NSGA-II',
colorDic = {'ABC1':'b', 'ABC1':'g','NSGAII':"darkorange", 'NSGAII-PC':'m', 'NSGAII-PM':'k', 'NSGAII-PMC':'r'} #NSGA-II',


dataList = [ "hm03","hm09g","yst04r","yst08r","dm01g","dm03g"]
motifLenList = [16,18,20,22]
repCount = 10
runCount = 20
genList = list(range(0, 1000, 40))
ref_point = [1.2, 1.2]
popSize = 200
sns.set(style="whitegrid")
sns.set_context("talk")

for data in dataList:
    for motLen in motifLenList:
        #df_algo['gen'] = genList
         for algo in algoList:
             gdf = pd.DataFrame(columns=['Similarity', 'Entropy'])
             #avg_hv = 0
             for runId in range(0, runCount):
                ldf = pd.read_csv(root + algoDirDic[algo] + '/Pareto front/' + data + '/' + str(motLen) + '/' + 'run' + str(runId) +'.txt', delimiter=r"\s+", header=None,
                        names=['Similarity', 'Entropy'], index_col=False)
                gdf = gdf.append(ldf, ignore_index=True, sort=True)
             # gdf = gdf.drop_duplicates()
             # gdf = gdf.reset_index(drop=True)
             # gdf['Similarity'] = gdf['Similarity'] * -1
             # gdf['Entropy'] = gdf['Entropy'] * -1
             # gdf = (df - df.min()) / (df.max() - df.min())
             mat = gdf.to_numpy()
             mat = np.multiply(mat, -1)
             I = NonDominatedSorting().do(mat, only_non_dominated_front=True)
             mat = mat[I]
             mat = np.multiply(mat, -1)
             plt.scatter(mat[:,0], mat[:,1], label=algo, facecolors='none', edgecolors=colorDic[algo], marker=markerDic[algo], linewidth=3)

         plt.xlabel('Similarity')
         plt.ylabel('Entropy')
         plt.legend(loc='upper center', bbox_to_anchor=(0.41, 1.08),
                   ncol=5, fancybox=True, prop={'size': 9.5})
         plt.suptitle('dataset='+data + ", len=" + str(motLen), fontsize=18)
         plt.show()
            # plt.clf()


     # plt.xlabel('Generation')
     # plt.ylabel('Hypervolume')
     # plt.legend()
     # # plt.show()
     # # plt.clf()
     # plt.savefig(outDir + data + '_' + str(motLen) +'_hv_progress.png', format='png', bbox_inches='tight')
     # plt.clf()
