#encoding=utf-8

import pandas as pd
import re
import scipy.stats as stats
import numpy as np
from Data import load_file
import time 
import datetime


dir='D:/kesci'
train_master_path='/data/train/train_master.csv'
test_master_path='/PPD-Second-Round-Data/复赛测试集/Kesci_Master_9w_gbk_1_test_set.csv'


#选取master文件中的部分category特征
category_list=['UserInfo_1','UserInfo_3','UserInfo_5','UserInfo_6','UserInfo_9','UserInfo_11','UserInfo_12','UserInfo_13','UserInfo_14','UserInfo_15',
              'UserInfo_16','UserInfo_17','UserInfo_21','UserInfo_22','Education_Info1','Education_Info2','Education_Info3','Education_Info4',
               'Education_Info5','Education_Info6','Education_Info7','Education_Info8','WeblogInfo_19','WeblogInfo_21','SocialNetwork_1','SocialNetwork_2',
              'SocialNetwork_7','SocialNetwork_12']

#对离散型变量进行二分处理
def getOneHot(category_data,category):
    temp_category_data=category_data[category]
    temp=pd.DataFrame()
    for i in category:
        temp_Series=temp_category_data[i].astype('category')
        temp=pd.concat([temp,temp_Series],axis=1)            
    temp=pd.get_dummies(temp,dummy_na=True)
    return temp

#获取train——master部分的特征
def get_train_master_category():
    train_master_data=load_file(dir,train_master_path)
    train_master_category_data=train_master_data[category_list]
    train_category_data=getOneHot(train_master_category_data,category_list)
    print(train_category_data.shape)
    return train_category_data

#获取test——master部分的特征
def get_test_master_category():
    test_master_data=load_file(dir,test_master_path)
    test_master_category_data=test_master_data[category_list]
    test_category_data=getOneHot(test_master_category_data,category_list)
    print(test_category_data.shape)
    return test_category_data
    
#确保特征值一致
def feature_check(train,test):
    list1=[]
    list2=[]
    for i in train.columns:
        list1.append(i)
    x=set(list1)
    for i in test.columns:
        list2.append(i)
    y=set(list2)
    diff=list(x-y)
    print(diff)
    new_train=train.drop(diff,axis=1)
    print(new_train.shape)
    print(test.shape)
    return new_train
    
    

if __name__=='__main__':
    train_master_category=get_train_master_category()
    test_master_category=get_test_master_category()
    new_train_master_category=feature_check(train_master_category,test_master_category)
    new_train_master_category.to_csv('D:/kesci/data/part_data/train_master_category.csv',index=None)
    test_master_category.to_csv('D:/kesci/data/part_data/test_master_category.csv',index=None)
    

   
    

    
