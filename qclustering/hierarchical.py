import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
from .dissimilarity import mmd, dissimilarity_matrix


def argmin(matrix):
    column_smallest = matrix.min().idxmin()
    row_smallest = matrix[column_smallest].idxmin()
    return row_smallest, column_smallest


# hierarchical visualization
def hierarchical_clustering(data: pd.DataFrame, qualitative_var: str, quantitative_vars=None, 
                            min_sample=30, dissimilarity=mmd, **dissimilarity_kwargs):
    if quantitative_vars is None:
        quantitative_vars = data.select_dtypes('number').columns
        
    # calculate first dissimilarity matrix
    dmatrix = dissimilarity_matrix(data=data, qualitative_var=qualitative_var, quantitative_vars=quantitative_vars, min_sample=min_sample, triangular=True, standardize=False, dissimilarity=dissimilarity, **dissimilarity_kwargs)
    
    # create list of initial clusters (1 per qualitative value)
    initial_clusters = pd.Series(dmatrix.index, index=range(len(dmatrix)))
    dmatrix.index = initial_clusters.index
    dmatrix.columns = initial_clusters.index
    
    # find the row and column of the minimum value in the 

    # initialize linkage matrix, later used for plotting the dendrogram 
    linkage = pd.DataFrame(columns=['child_1','child_2','distance','size','qualitative_values'])
    linkage.index.name = 'cluster_id'

    # 
    while len(dmatrix) > 1:
        # new cluster id
        new_cluster_id = len(initial_clusters) + len(linkage)

        # find 2 child linkage whose dissimilarity is minimum
        child_1_id, child_2_id = argmin(dmatrix)
        new_distance = dmatrix.loc[child_1_id, child_2_id]

        # clean up dmatrix
        dmatrix = dmatrix.drop(index=[child_1_id, child_2_id])
        dmatrix = dmatrix.drop(columns=[child_1_id, child_2_id])
        dmatrix.loc[new_cluster_id, :] = np.nan
        dmatrix.loc[:, new_cluster_id] = np.nan

        # concatenate qualitative_values
        child_1_qualitative_values = [initial_clusters.iloc[child_1_id]] if child_1_id in initial_clusters.index else linkage.loc[child_1_id, 'qualitative_values']
        child_2_qualitative_values = [initial_clusters.iloc[child_2_id]] if child_2_id in initial_clusters.index else linkage.loc[child_2_id, 'qualitative_values']
        new_qualitative_values = child_1_qualitative_values + child_2_qualitative_values

        # identify instances of the new cluster
        new_cluster_mask = data[qualitative_var].isin(new_qualitative_values)
        new_cluster_instances = data[new_cluster_mask][quantitative_vars]
        
        # sum size
        new_size = len(new_cluster_instances)

        # update the dissimilarity matrix with new distances to the new cluster
        for c in dmatrix.columns[:-1]:
            qualitative_values = [initial_clusters.iloc[c]] if c in initial_clusters.index else linkage.loc[c, 'qualitative_values']
            cluster_mask = data[qualitative_var].isin(qualitative_values)
            cluster_instances = data[cluster_mask][quantitative_vars]
            dmatrix.loc[new_cluster_id, c] = dissimilarity(new_cluster_instances, cluster_instances)

        # add cluster to linkage dataframe
        linkage.loc[new_cluster_id, :] = [child_1_id, child_2_id, new_distance, new_size, new_qualitative_values]
    
    return initial_clusters, linkage


def plot_dendrogram(initial_clusters, linkage, figsize=(8.5, 12), ax=None, **kwargs):
    linkage = linkage.iloc[:,:4].astype(float)
    linkage.loc[:'distance'] += 0.001
    labels = [c+' ('+str(i)+')' for i,c in enumerate(initial_clusters)]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    dend = shc.dendrogram(linkage, orientation='right', labels=labels, distance_sort=True, ax=ax, **kwargs)
    ax.set_xlabel('Dissimilarity');
    return ax

