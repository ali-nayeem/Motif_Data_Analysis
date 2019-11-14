import pandas as pd
#import matplotlib.pyplot as plt
#from patsy import dmatrices
import numpy as np
#import statsmodels.api as sm
import matplotlib.pyplot as plt
#from scipy.stats import gaussian_kde
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#from sklearn.decomposition import PCA
import os
from shutil import copyfile

root = '../Data/NSGA-DM/'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(root):
    for file in f:
        part_list = file.split("\\")
        path = root + 'output/'
        for part in part_list:
            if '.txt' in part:
                copyfile(root+file, path+part)

            else:
                path = path + part + '/'
                if not os.path.exists(path):
                    os.makedirs(path)

