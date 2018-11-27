#!/usr/bin/env python
# coding: utf-8

import numpy as np
np.set_printoptions(threshold=np.nan)
import csv
import scipy.stats as stats


file = open("pa3_train_reduced.csv","r")


csv_content = np.array([])  
#create an array to store the positions in the sorted subtable where the values
indices_where_change = np.zeros([])


csv_content = np.genfromtxt("pa3_train_reduced.csv",delimiter = ",")
y_orig = csv_content[...,0:1]
y_orig[y_orig==3.0]=1
y_orig[y_orig==5.0]=-1
# print(y_orig)
c = 0

#maintain an array for Benefit values for all features
benefit_array_for_one_feature = np.zeros(101)
index_of_array_after_sorting_for_max_bene = np.zeros(101)



def gini_data(y):
    #count the Y equal to one
    c_root_pos=np.count_nonzero(y==1)
    c_root_total = len(y)
    return ( 1- (c_root_pos/c_root_total)**2 -((c_root_total-c_root_pos)/c_root_total)**2 )







def get_split(X, y_to_take):

    for col in range(X.shape[1]):
        single_feature = csv_content[..., col + 1]

        #stack y_to_take and X[col]
        stacked = np.column_stack((y_orig, single_feature))
        # pass as arg the indices of the sorted 2nd column
        sorted = stacked[np.argsort(stacked[:, 1])]

        y_to_take = sorted[..., 0]

        indices_where_change = np.where(y_to_take[:-1] != y_to_take[1:])[0]
        indices_where_Actual_thresholds_are = indices_where_change  # print("y to take \n"+str(y_to_take))


        #Ashish done#####
        for i in range(len(indices_where_Actual_thresholds_are)):
            CL_pos = np.count_nonzero(y_to_take[:indices_where_Actual_thresholds_are[i] + 1] == 1)

            CL_total = len(y_to_take[:indices_where_Actual_thresholds_are[i] + 1])
            CL_neg = CL_total - CL_pos

            p_plus = CL_pos / (CL_pos + CL_neg)
            p_neg = CL_neg / (CL_neg + CL_pos)
            UAL = 1 - (p_plus) ** 2 - (p_neg) ** 2

            CR_pos = np.count_nonzero(y_to_take[indices_where_Actual_thresholds_are[i] + 1:] == 1)
            CR_total = len(y_to_take[indices_where_Actual_thresholds_are[i] + 1:])
            CR_neg = CR_total - CR_pos
            p_plus_r = CR_pos / (CR_pos + CR_neg)
            p_neg_r = CR_neg / (CR_neg + CR_pos)
            UAR = 1 - (p_plus_r) ** 2 - (p_neg_r) ** 2

            pl = (CL_total) / len(y_to_take)
            pr = (CR_total) / len(y_to_take)

            Benefit_of_split_at_this_i = gini_data(y_to_take) - pl * UAL - pr * UAR

            if (Benefit_of_split_at_this_i > benefit_array_for_one_feature[col + 1]):
                benefit_array_for_one_feature[col + 1] = Benefit_of_split_at_this_i
                index_of_array_after_sorting_for_max_bene[col + 1] = i

            #select the attribute(feature) that give the highest benefit
            # AND the index
            #use max
            attribute_max_benefit=np.argmax(benefit_array_for_one_feature[1:])
            value_max_benefit=index_of_array_after_sorting_for_max_bene[attribute_max_benefit+1]

            #assign group above and below the index

    return




def split(max_benefit_feature_index,threshold):
    threshold=int(threshold)
    data = csv_content[...,1:]
    x_feature_we_need= data[...,max_benefit_feature_index]

    sorted_table = data[np.argsort(data[:, max_benefit_feature_index])]
    #print the table to see if sorted or not

    sorted_X = np.sort(x_feature_we_need)
    value_which_comes_before_split = sorted_X[threshold]
    print("threshold x after which we split is "+str(value_which_comes_before_split))
    left_table = sorted_table[:threshold+1,...]
    right_table = sorted_table[threshold:,...]
    return  left_table,right_table


max_benefit_feature_index,index_threshold = get_split(csv_content[..., 1:], y_orig)
left_table,right_table=split(max_benefit_feature_index,index_threshold)




