from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np


def scale(dataFrame):
    result = dataFrame.copy()
    for feature_name in dataFrame.columns:
        if feature_name != 'Class':
            max_value = dataFrame[feature_name].max()
            min_value = dataFrame[feature_name].min()
            result[feature_name] = (
                dataFrame[feature_name] - min_value) / (max_value - min_value)
        else:
            result[feature_name] = dataFrame[feature_name]
    return result


def calculate_stability(ind, df, sidx, max_k):
    indexes = sidx[:, ind]
    indexes = np.delete(indexes, np.where(indexes == ind))
    # https://stackoverflow.com/a/69782015/232017
    # class objects count in neighbour
    d = (df["Class"][ind] == df["Class"][indexes]).cumsum()
    # neighbour count sequence
    j = np.arange(1, len(d) + 1)
    # get dominant neighbour count:
    crit_objs = j[(d / j > 1 / 2)]
    # calculate stability
    result = crit_objs[-1] / max_k if len(crit_objs) else 0
    return result


df = pd.read_csv("Dry_Bean.txt", sep='\t')

print("Do you want scale data frame? ([y]/n)", end='')
answer = input()
if not answer or answer.lower() == 'y':
    df = scale(df)
    print("\t scaled...")

# Findinx max k
print('Finding class values with objects count...')
print(df['Class'].value_counts())
# -3 because object cannot be neighbour to itself
max_k = 2 * df['Class'].value_counts().min() - 3
print(f'Max k = {max_k}')

print('Finding distances between all objects...')
dist_sq = euclidean_distances(df.iloc[:, :-1]).round(6)
print(dist_sq)
# np.savetxt("result_00_distances.txt", dist_sq, delimiter='\t', fmt='%.6f')

print('Finding sequence of neighbours...')
sidx = np.argsort(dist_sq, axis=0)[:max_k+1, :]
# np.savetxt("result_01_sequence_indexes.txt",
#    sidx, delimiter='\t', fmt='%d')

print('Finding stability of objects...')
stability = [calculate_stability(x, df, sidx, max_k)
             for x in range(df.shape[0])]
np.savetxt("result_03_stability.txt", stability,
           delimiter='\t', fmt='%.6f')
