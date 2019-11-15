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
outDir = '../Output/'
columns = [ 'Method', 'Dataset', 'Motif Length','Similarity', 'Entropy']  # ['Entropy', 'TC', 'Gap', 'SimG', 'SimNG', 'GapCon']
algoList = [ 'NSGAII', 'NSGAII-PM', 'NSGAII-PC', 'NSGAII-PMC']
algoDirDic = {'ABC-E':'ABC', 'ABC-S':'ABC','NSGAII':'NSGA-DefaultMutation-DefaultCrossover', 'NSGAII-PC':'NSGA-DefaultMutation-MyCrossover', 'NSGAII-PM':'NSGA-MyMutation-DefaultCrossover', 'NSGAII-PMC':'NSGA-MyMutation-MyCrossover'} #NSGA-II',
markerDic = {'ABC-S':'+', 'ABC-E':'x','NSGAII':"s", 'NSGAII-PC':'*', 'NSGAII-PM':'o', 'NSGAII-PMC':'^'} #NSGA-II',
colorDic = {'ABC-S':'b', 'ABC-E':'c','NSGAII':'g', 'NSGAII-PC':'m', 'NSGAII-PM':'k', 'NSGAII-PMC':'r'} #NSGA-II',


dataList = [ "hm03","hm09g","yst04r","yst08r","dm01g","dm03g"]
motifLenList = [16,18,20,22]
motifLenDic = {16:0,18:1,20:2,22:3}
dataDic = { "hm03":0,"hm09g":1,"yst04r":2,"yst08r":3,"dm01g":4,"dm03g":5}
toolList = ['ABC-S', 'ABC-E']
toolFitnessDic= {'ABC-S':'Similarity', 'ABC-E':'Entropy'}
repCount = 10
runCount = 20
genList = list(range(0, 1000, 40))
ref_point = [1.2, 1.2]
popSize = 200
# sns.set(style="whitegrid")
# sns.set_context("talk")

nr_rows = 6
nr_cols = 4

fig, axs = plt.subplots(nr_rows, nr_cols, squeeze=False,
                        figsize=(nr_cols * 3.5, nr_rows * 2.5))  # , figsize=(nr_cols * 4, nr_rows * 3)
tdf = {}
for data in dataList:
    tdf['ABC-S'] = pd.read_csv(root + 'ABC/Best Outputs/' + 'Similarity' + '/' + data + '.txt.txt',
                      delimiter=r"\s+", header=None,
                      names=['Motif Length', 'Similarity', 'Entropy'], index_col=False)

    tdf['ABC-E'] = pd.read_csv(root + 'ABC/Best Outputs/' + 'Entropy' + '/' + data + '.txt.txt',
                               delimiter=r"\s+", header=None,
                               names=['Motif Length', 'Similarity', 'Entropy'], index_col=False)
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
             axs[dataDic[data]][motifLenDic[motLen]].scatter(mat[:,0], mat[:,1], label=algo, facecolors='none', edgecolors=colorDic[algo], marker=markerDic[algo], linewidth=2)
         for tool in toolList:
             ldf = tdf[tool]
             ldf = ldf[ ldf['Motif Length'] == motLen]
             mat = ldf[['Similarity', 'Entropy']].to_numpy()
             mat = np.multiply(mat, -1)
             I = NonDominatedSorting().do(mat, only_non_dominated_front=True)
             mat = mat[I]
             mat = np.multiply(mat, -1)
             axs[dataDic[data]][motifLenDic[motLen]].scatter(mat[:,1], mat[:,0], label=tool, edgecolors=colorDic[tool], marker=markerDic[tool], linewidth=2) #facecolors='none'

         # plt.xlabel('Similarity')
         # plt.ylabel('Entropy')
         # plt.legend(loc='upper center', bbox_to_anchor=(0.41, 1.08),
         #           ncol=5, fancybox=True, prop={'size': 9.5})
         # plt.suptitle('dataset='+data + ", len=" + str(motLen), fontsize=18)
         axs[dataDic[data]][motifLenDic[motLen]].set_title('dataset='+data + ", len=" + str(motLen), fontsize=12)
         axs[dataDic[data]][motifLenDic[motLen]].grid(b=True, which='major', color='grey', linewidth=0.5, axis='y')
         axs[dataDic[data]][motifLenDic[motLen]].grid(b=True, which='major', color='grey', linewidth=0.5, axis='x')

         axs[5][motifLenDic[motLen]].set_xlabel('Entropy', fontsize=12)
    axs[dataDic[data]][0].set_ylabel('Similarity', fontsize=12)
handles, labels = axs[0][0].get_legend_handles_labels()
fig.tight_layout()
fig.legend(handles, labels, loc='best', ncol=6, fontsize=12)
fig.subplots_adjust(wspace=0.18, hspace=0.3)
fig.savefig(outDir + 'all_pf.png', format='png', bbox_inches='tight') #data + '_' +

#plt.show()
            # plt.clf()


     # plt.xlabel('Generation')
     # plt.ylabel('Hypervolume')
     # plt.legend()
     # # plt.show()
     # # plt.clf()
     # plt.savefig(outDir + data + '_' + str(motLen) +'_hv_progress.png', format='png', bbox_inches='tight')
     # plt.clf()
