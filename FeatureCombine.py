#encoding='Utf-8'
import pandas as pd 

#组合特征:feature_plus
def feature_Plus(train_master_numeric_data,featureList,eps):
    feature_plus_list=[]
    for i in featureList:
        for j in featureList:
            if i!='target' and j!='target' and i<j:
                i_plus_j=train_master_numeric_data[i]+train_master_numeric_data[j]
                if(abs(train_master_numeric_data['target'].corr(i_plus_j))>eps):
                    temp=(i,j,abs(train_master_numeric_data['target'].corr(i_plus_j)))
                    feature_plus_list.append(temp)
    print(len(feature_plus_list))
    return feature_plus_list
    
    
#相乘
def feature_Mul(train_master_numeric_data,featureList,eps):
    feature_mul_list=[]
    for i in featureList:
        for j in featureList:
            if i!='target' and j!='target' and i<j:
                i_mul_j=train_master_numeric_data[i]*train_master_numeric_data[j]
                if(abs(train_master_numeric_data['target'].corr(i_mul_j))>eps):
                    temp=(i,j,abs(train_master_numeric_data['target'].corr(i_mul_j)))
                    feature_mul_list.append(temp)
    print(len(feature_mul_list))
    return feature_mul_list

#相除
def feature_Div(train_master_numeric_data,featureList,eps):
    feature_div_list=[]
    for i in featureList:
        for j in featureList:
            if i!='target' and j!='target' and i<j:
                i_div_j=train_master_numeric_data[i]/train_master_numeric_data[j]
                if(abs(train_master_numeric_data['target'].corr(i_div_j))>eps):
                    temp=(i,j,abs(train_master_numeric_data['target'].corr(i_div_j)))
                    feature_div_list.append(temp)
    print(len(feature_div_list))
    return feature_div_list
    

#feature_minus
def feature_Minus(train_master_numeric_data,featureList,eps):
    feature_minus_list=[]
    for i in featureList:
        for j in featureList:
            if i!='target' and j!='target' and i<j:
                i_minus_j=train_master_numeric_data[i]-train_master_numeric_data[j]
                if(abs(train_master_numeric_data['target'].corr(i_minus_j))>eps):
                    temp=(i,j,abs(train_master_numeric_data['target'].corr(i_minus_j)))
                    feature_minus_list.append(temp)
    print(len(feature_minus_list))
    return feature_minus_list




