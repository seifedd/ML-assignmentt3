"""
Gini function implementation to minimize the enthropy
"""
import data
import numpy as np


def Gini_data(Y):
    #count the Y equal to one
    c_pos=len(np.where(Y == 1)[0])/len(Y)
    #count the Y equal to 0
    c_neg=len(np.where(Y == -1)[0])/len(Y)
    return 1-c_pos**2-c_neg**2

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

    UAR=1-((CR_pos/(CR_pos+CR_neg))**2)-((CR_neg/(CR_neg+CR_pos))**2)

    # Gini index for the right branch
    below_threshold_indices = X < Threshlod
    CL_pos=np.count_nonzero(Y[below_threshold_indices]==1)
    CL_neg=np.count_nonzero(Y[below_threshold_indices]==-1)

    #calculate the Gini for the  left brnch

    UAL=1-(CL_pos/(CL_pos+CL_neg))**2-(CL_neg/(CL_neg+CL_pos))**2

    UA=Gini_data(Y)

    pl=(CL_pos+CL_neg)/len(Y)
    pr=(CR_pos+CR_neg)/len(Y)

    #benefit of split for feature f_i
    B=UA-pl*UAL-pr*UAR

    return B


def threshold_func(X,Y):
   # pass   as arg the indices of the sorted 2nd column
    YX=np.column_stack((Y, X[:,0]))
    sorted = YX[np.argsort(YX[:, 1])]


    y1, x1 = np.hsplit(sorted, 2)
    indices_where_change = np.where(y1[:-1] != y1[1:])[0]
    return  X[indices_where_change]


[Y,X]=data.open_csv("pa3_train_reduced")

thresh=threshold_func(X,Y)
print(len(thresh))
for th  in thresh:
    B=Gini(X[:,0],Y,th)
    print(B)
#
# for thress in thresh:
#     B=Gini(X[:,1],Y,thress)
#     print(B)



