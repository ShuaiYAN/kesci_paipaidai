#encoding=utf-8

import pandas as pd
import re
import scipy.stats as stats
import numpy as np
from Data import load_file
import time 
import datetime


dir='D:/kesci'
train_logInfo_path='/data/train/train_LogInfo.csv'
test_logInfo_path='/PPD-Second-Round-Data/复赛测试集/LogInfo_9w_1.csv'


#类别特征处理
def getOneHot(category_data,category):
    temp_category_data=category_data[category]
    temp=pd.DataFrame()
    for i in category:
        temp_Series=temp_category_data[i].astype('category')
        temp=pd.concat([temp,temp_Series],axis=1)            
    temp=pd.get_dummies(temp,dummy_na=True)
    return temp

#日期特征处理
def get_day(date_col_1,date_col_2):
    date1=[time.strptime(i,'%Y-%m-%d') for i in date_col_1]
    date2=[time.strptime(i,'%Y-%m-%d') for i in date_col_2]
    date1=[datetime.datetime(date1[i][0],date1[i][1],date1[i][2]) for i in range(len(date1))]
    date2=[datetime.datetime(date2[i][0],date2[i][1],date2[i][2]) for i in range(len(date2))]
    d=[(date1[i]-date2[i]).days for i in range(len(date1))]
    return d  
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

def get_train_logInfo():
    train_LogInfo=load_file(dir,train_logInfo_path)
    train_LogInfo_Idx=train_LogInfo['Idx']
    category=['LogInfo1','LogInfo2']
    train_LogInfo_1=getOneHot(train_LogInfo,category)
    train_LogInfo_1=pd.concat([train_LogInfo_Idx,train_LogInfo_1],axis=1)
    print('类别特征转换成哑变量')
    print(train_LogInfo_1.shape)
    #数据聚合
    train_LogInfo_1_GroupBy=train_LogInfo_1.groupby(train_LogInfo_1['Idx'])
    train_LogInfo_1=train_LogInfo_1_GroupBy.aggregate(np.sum)
    print('按Idx聚合数据')
    print(train_LogInfo_1.shape)
    Update_day=get_day(train_LogInfo['Listinginfo1'],train_LogInfo['LogInfo3'])
    Update_day=pd.Series(Update_day,name='Update_day')
    train_UserUpdate_day=pd.concat([train_LogInfo_Idx,Update_day],axis=1)
    train_UserUpdate_day_GroupBy=train_UserUpdate_day.groupby(train_UserUpdate_day['Idx'])
    train_LogInfo_2=train_UserUpdate_day_GroupBy.aggregate(np.mean)
    #日期型特征提取
    print(train_LogInfo_2.shape)
    train_Idx_counts=train_LogInfo_Idx.value_counts(sort=False)
    train_counts=pd.Series(train_Idx_counts,name='Idx_counts')
    train_Idx=pd.Series(train_Idx_counts.index,name='Idx',index=train_Idx_counts.index)
    train_Idx_counts=pd.concat([train_Idx,train_counts],axis=1)
    train_Idx_counts=pd.DataFrame(train_Idx_counts)
    #出现次数统计
    print(train_Idx_counts.shape)
    train_LogInfo=train_Idx_counts.join(train_LogInfo_1,how='left')
    train_LogInfo=train_LogInfo.join(train_LogInfo_2,how='left')
    #数据合并，生成最终数据
    print(train_LogInfo.shape)
    return train_LogInfo

def get_test_logInfo():
    test_LogoInfo=load_file(dir,test_logInfo_path)
    test_LogoInfo_Idx=test_LogoInfo['Idx']
    category=['LogInfo1','LogInfo2']
    test_LogoInfo_1=getOneHot(test_LogoInfo,category) 
    test_LogoInfo_1=pd.concat([test_LogoInfo_Idx,test_LogoInfo_1],axis=1)
    print(test_LogoInfo_1.shape)
    #数据聚合
    test_LogoInfo_1_GroupBy=test_LogoInfo_1.groupby(test_LogoInfo_1['Idx'])
    test_LogoInfo_1=test_LogoInfo_1_GroupBy.aggregate(np.sum)
    print(test_LogoInfo_1.shape)
    test_Update_day=get_day(test_LogoInfo['Listinginfo1'],test_LogoInfo['LogInfo3'])
    test_Update_day=pd.Series(test_Update_day,name='Update_day')
    test_UserUpdate_day=pd.concat([test_LogoInfo_Idx,test_Update_day],axis=1)
    print(test_UserUpdate_day.shape)
    test_UserUpdate_day_GroupBy=test_UserUpdate_day.groupby(test_UserUpdate_day['Idx'])
    test_UserInfo_2=test_UserUpdate_day_GroupBy.aggregate(np.mean)
    print(test_UserInfo_2.shape)
    test_Idx_counts=test_LogoInfo_Idx.value_counts(sort=False)
    test_counts=pd.Series(test_Idx_counts,name='Idx_counts')
    test_Idx=pd.Series(test_Idx_counts.index,name='Idx',index=test_Idx_counts.index)
    test_Idx_counts=pd.concat([test_Idx,test_counts],axis=1)
    test_Idx_counts=pd.DataFrame(test_Idx_counts)
    print(test_Idx_counts.shape)
    test_LogoInfo=test_Idx_counts.join(test_LogoInfo_1,how='left')
    test_LogoInfo=test_LogoInfo.join(test_UserInfo_2,how='left')
    print(test_LogoInfo.shape)
    return test_LogoInfo

if __name__=='__main__':
    train_logInfo=get_train_logInfo()
    test_logInfo=get_test_logInfo()
    new_train_LogInfo=feature_check(train_logInfo,test_logInfo)
    new_train_LogInfo.to_csv('D:/kesci/data/part_data/train_LogInfo.csv',index=None)
    test_logInfo.to_csv('D:/kesci/data/part_data/test_LogInfo.csv',index=None)
    

