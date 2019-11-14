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
algoList = ['ABC', 'NSGAII', 'NSGAII-PM', 'NSGA-II-PC', 'NSGAII-PMC']
algoDirDic = {'ABC':'ABC', 'NSGAII':'NSGA-DefaultMutation-DefaultCrossover', 'NSGA-II-PC':'NSGA-DefaultMutation-MyCrossover', 'NSGAII-PM':'NSGA-MyMutation-DefaultCrossover', 'NSGAII-PMC':'NSGA-MyMutation-MyCrossover'} #NSGA-II',
dataList = ["hm03","hm09g","yst04r","yst08r","dm01g","dm03g"]

repCount = 10
runCount = 10
treeErrorCol = 1  # FN Rate
df = pd.DataFrame(columns=columns)


