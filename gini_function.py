"""
Gini function implementation to minimize the enthropy
"""
import data
import numpy as np

def Gini(X,Y,Threshlod):

    #sort X based on x values ascendingly
    X.sort()
    b = np.argsort(X)
    Y=Y[b]

    #Gini index for the Right branch
    above_threshold_indices = X >= Threshlod
    CR_pos=np.count_nonzero(Y[above_threshold_indices]==1)
    CR_neg=np.count_nonzero(Y[above_threshold_indices]==-1)
    #calculate the Gini for the  right brnch
    p_plus=CR_pos/(CR_pos+CR_neg)
    p_neg=CR_neg/(CR_neg+CR_pos)
    UAR=1-p_plus^2-p_neg^2

    # Gini index for the right branch
    below_threshold_indices = X < Threshlod
    CL_pos=np.count_nonzero(Y[below_threshold_indices]==1)
    CL_neg=np.count_nonzero(Y[below_threshold_indices]==-1)

    #calculate the Gini for the  left brnch
    p_plus=CL_pos/(CL_pos+CL_neg)
    p_neg=CL_neg/(CL_neg+CL_pos)
    UAL=1-p_plus^2-p_neg^2

    



[labels,features]=data.open_csv("pa3_train_reduced")
print(Gini(features[:,1],labels,0.1))