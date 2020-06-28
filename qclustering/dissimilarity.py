import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_kernels


# MMD distance function
def mmd(X, Y, biased=False, gamma=None):
    # based on implementation from: https://github.com/emanuele/kernel_two_sample_test
    # pvalue bug fix: https://github.com/smkia/kernel_two_sample_test/commit/601725d7abb60b2b7f258d702b76ac6c80785aee
    
    
    # code from functon above
    m = len(X)
    n = len(Y)
    
    # stack all instances together
    XY = np.vstack([X, Y])
    
    # compute the kernel matrix using RBF kernel function
    K = pairwise_kernels(XY, metric='rbf', gamma=gamma)
    KX = K[:m, :m]
    KY = K[m:, m:]
    KXY = K[:m, m:]
    
    if biased:
        # compute the MMD² biased statistic
        mmd2 = 1.0 / (m * m) * (KX.sum() - KX.diagonal().sum()) + \
               1.0 / (n * n) * (KY.sum() - KY.diagonal().sum()) - \
               2.0 / (m * n) * KXY.sum()
    else:
        # compute the MMD² unbiased statistic
        mmd2 = 1.0 / (m * (m - 1)) * (KX.sum() - KX.diagonal().sum()) + \
               1.0 / (n * (n - 1)) * (KY.sum() - KY.diagonal().sum()) - \
               2.0 / (m * n) * KXY.sum()
        
    # return the sqrt of the squared MMD measure
    return 0.0 if mmd2 < 0 else np.sqrt(mmd2)


# dissimilarity matrix
def dissimilarity_matrix(data: pd.DataFrame, qualitative_var: str, quantitative_vars=None, min_sample=30, triangular=False, standardize=True, dissimilarity=mmd, **dissimilarity_kwargs):
    """
    
    """
    if quantitative_vars is None:
        quantitative_vars = data.select_dtypes('number').columns

    # get qualitative values that have sufficient sample size
    qualitative_values = data.loc[:, qualitative_var].value_counts()
    qualitative_values = qualitative_values[qualitative_values >= min_sample].index
    
    # dissimilarity matrix
    dmatrix = pd.DataFrame(index=qualitative_values, columns=qualitative_values, dtype=float)
    for i in range(len(qualitative_values)):
        # diagonal
        if not triangular:
            dmatrix.iloc[i,i] = 0
        # get sample i
        mask_i = (data[qualitative_var]==qualitative_values[i])
        sample_i = data[mask_i][quantitative_vars]
        for j in range(i+1,len(qualitative_values)):
            # get sample j
            mask_j = (data[qualitative_var]==qualitative_values[j])
            sample_j = data[mask_j][quantitative_vars]
            # set value on both upper and lower triangle of the matrix
            dmatrix.iloc[i,j] = dissimilarity(sample_i, sample_j, **dissimilarity_kwargs)
            if not triangular:
                dmatrix.iloc[j,i] = dmatrix.iloc[i,j]
    
    return dmatrix

