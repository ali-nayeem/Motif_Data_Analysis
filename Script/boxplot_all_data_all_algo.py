import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

root = '../Data/'
outDir = '../Output/'
columns = [ 'Method', 'Dataset', 'Motif Length','Similarity', 'Entropy']  # ['Entropy', 'TC', 'Gap', 'SimG', 'SimNG', 'GapCon']
algoList = [ 'NSGAII', 'NSGAII-PM', 'NSGAII-PC', 'NSGAII-PMC']
algoDirDic = {'ABC':'ABC', 'NSGAII':'NSGA-DefaultMutation-DefaultCrossover', 'NSGAII-PC':'NSGA-DefaultMutation-MyCrossover', 'NSGAII-PM':'NSGA-MyMutation-DefaultCrossover', 'NSGAII-PMC':'NSGA-MyMutation-MyCrossover'} #NSGA-II',
toolList = ['ABC']
dataList = ["hm03","hm09g","yst04r","yst08r","dm01g","dm03g"]
motifLenList = [16, 18, 20, 22]
repCount = 10
runCount = 20
simCol = 0
entrpCol = 1
toolFitnessList = ['Similarity', 'Entropy']

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
   # for fitness in toolFitnessList:
        ldf = pd.read_csv(root + tool + '/Best Outputs/' + 'Similarity' + '/' + data + '.txt.txt', delimiter=r"\s+", header=None,
                    names=['Motif Length', 'Similarity'], index_col=False)

        ldf['Method'] = tool
        ldf['Dataset'] = data
        entropy = np.genfromtxt(root + tool + '/Best Outputs/' + 'Entropy' + '/' + data + '.txt.txt',
                           delimiter=r"\s+")
        ldf['Entropy'] = entropy
        gdf = gdf.append(ldf, ignore_index=True, sort=True)
    sns.set(style="whitegrid")
    sns.set_context("talk")
    # plt.figure(figsize=(20,3))
    # sns.set(style="whitegrid")
    # g = sns.boxplot(x="Motif Length", y="Similarity",
    #                 hue="Method", data=gdf, linewidth=1.5) #, linewidth=1
    g = sns.boxplot(x="Motif Length", y="Entropy",
                    hue="Method", data=gdf, linewidth=1.5)  # , linewidth=1
    g.legend_.remove()
    plt.legend(loc='upper center', bbox_to_anchor=(0.41, 1.08),
              ncol=5, fancybox=True, prop={'size': 9.5}) #prop={'size': 8}, , shadow=True   fontsize='small'
    plt.savefig(outDir + 'Entropy/' +data + '_entropy_all_boxplot.png', format='png', bbox_inches='tight')
    plt.clf()
    #plt.show()
