import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from .dissimilarity import mmd, dissimilarity_matrix

# 2d visualization
def plot_dissimilarity_2d(data: pd.DataFrame, qualitative_var: str, quantitative_vars=None, min_sample=30, dissimilarity=mmd, ax=None, figsize=(14,12), s=1000, c='#00FF0033', **dissimilarity_kwargs):
    
    dmatrix = dissimilarity_matrix(data=data, qualitative_var=qualitative_var, quantitative_vars=quantitative_vars, min_sample=min_sample, dissimilarity=dissimilarity, **dissimilarity_kwargs)

    # dimensionality reduction
    pca = PCA(2).fit(dmatrix)
    dmatrix_2d = pd.DataFrame(pca.transform(dmatrix), 
                              columns=['PCA0', 'PCA1'], 
                              index=dmatrix.index)
    explained_variance_pct = pca.explained_variance_ratio_[:1].sum()
    
    # visualization
    ax = dmatrix_2d.plot('PCA0', 'PCA1', kind='scatter', ax=ax, figsize=figsize, s=s, c=c)
    for k, v in dmatrix_2d.iterrows():
        ax.annotate(k, v, ha='center', va='center', fontsize='9')
    
    return ax, explained_variance_pct;