#encoding=utf-8

import pandas as pd
import re
import scipy.stats as stats
import numpy as np
from Data import load_file
import time 
import datetime


dir='D:/kesci'
train_UserUpdate_path='/data/train/train_UserUpdateInfo.csv'
test_UserUpdate_path='/PPD-Second-Round-Data/复赛测试集/Userupdate_Info_9w_1.csv'


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
    date1=[time.strptime(i,'%Y/%m/%d') for i in date_col_1]
    date2=[time.strptime(i,'%Y/%m/%d') for i in date_col_2]
    date1=[datetime.datetime(date1[i][0],date1[i][1],date1[i][2]) for i in range(len(date1))]
    date2=[datetime.datetime(date2[i][0],date2[i][1],date2[i][2]) for i in range(len(date2))]
    d=[(date1[i]-date2[i]).days for i in range(len(date1))]
    return d  

#获取train部分的UserUpdate特征
def get_train_UserUpdate():
    train_UserUpdate=load_file(dir,train_UserUpdate_path)
    train_update_Idx=train_UserUpdate['Idx']
    #对UserupdateInfo1进行哑变量转换
    category=['UserupdateInfo1']
    train_UserUpdateInfo1=getOneHot(train_UserUpdate,category)
    #连接上Idx标识
    train_UserUpdateInfo_1=pd.concat([train_update_Idx,train_UserUpdateInfo1],axis=1)
    print(train_UserUpdateInfo_1.shape)
    train_UserUpdateInfo_GroupBy=train_UserUpdateInfo_1.groupby(train_UserUpdateInfo_1['Idx'])
    train_UserInfo_1=train_UserUpdateInfo_GroupBy.aggregate(np.sum)
    print('UserUpdateInfo1的特征处理')
    print(train_UserInfo_1.shape)
    #对日期特征进行处理
    Update_day=get_day(train_UserUpdate['ListingInfo1'],train_UserUpdate['UserupdateInfo2'])
    Update_day=pd.Series(Update_day,name='Update_day')
    train_UserUpdate_day=pd.concat([train_update_Idx,Update_day],axis=1)
    train_UserUpdate_day_GroupBy=train_UserUpdate_day.groupby(train_UserUpdate_day['Idx'])
    train_UserInfo_2=train_UserUpdate_day_GroupBy.aggregate(np.mean)
    print('UserupdateInfo2的特征处理')
    print(train_UserInfo_2.shape)
    #第三部分，对UserId出现次数的统计
    train_Idx_counts=train_update_Idx.value_counts(sort=False)
    train_counts=pd.Series(train_Idx_counts,name='Idx_counts')
    train_Idx=pd.Series(train_Idx_counts.index,name='Idx',index=train_Idx_counts.index)
    train_Idx_counts=pd.concat([train_Idx,train_counts],axis=1)
    train_Idx_counts=pd.DataFrame(train_Idx_counts)
    print('用户update次数的统计')
    print(train_Idx_counts.shape)
    train_Update=train_Idx_counts.join(train_UserInfo_1,how='left')
    train_Update=train_Update.join(train_UserInfo_2,how='left')
    print('train_Userupdate最终的数据输出')
    print(train_Update.shape)
    return train_Update

def get_test_UserUpdate():
    test_UserUpdate=load_file(dir,test_UserUpdate_path)
    test_update_Idx=test_UserUpdate['Idx']
    category=['UserupdateInfo1']
    test_UserUpdateInfo1=getOneHot(test_UserUpdate,category)
    test_UserUpdateInfo_1=pd.concat([test_update_Idx,test_UserUpdateInfo1],axis=1)
    print(test_UserUpdateInfo_1.shape)
    test_UserUpdateInfo_GroupBy=test_UserUpdateInfo_1.groupby(test_UserUpdateInfo_1['Idx'])
    test_UserInfo_1=test_UserUpdateInfo_GroupBy.aggregate(np.sum)
    print('test_UserUpdateInfo1的处理')
    print(test_UserInfo_1.shape)
    test_Update_day=get_day(test_UserUpdate['ListingInfo1'],test_UserUpdate['UserupdateInfo2'])
    test_Update_day=pd.Series(test_Update_day,name='Update_day')
    test_UserUpdate_day=pd.concat([test_update_Idx,test_Update_day],axis=1)
    test_UserUpdate_day_GroupBy=test_UserUpdate_day.groupby(test_UserUpdate_day['Idx'])
    test_UserInfo_2=test_UserUpdate_day_GroupBy.aggregate(np.mean)
    print("test_userInfo2的处理")
    print(test_UserInfo_2.shape)
    test_Idx_counts=test_update_Idx.value_counts(sort=False)
    test_counts=pd.Series(test_Idx_counts,name='Idx_counts')
    test_Idx=pd.Series(test_Idx_counts.index,name='Idx',index=test_Idx_counts.index)
    test_Idx_counts=pd.concat([test_Idx,test_counts],axis=1)
    test_Idx_counts=pd.DataFrame(test_Idx_counts)
    print('test用户修改记录统计')
    print(test_Idx_counts.shape)
    test_Update=test_Idx_counts.join(test_UserInfo_1,how='left')
    test_Update=test_Update.join(test_UserInfo_2,how='left')
    print("testUserUpdate最终数据框")
    print(test_Update.shape)
    return test_Update

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
    train_UserUpdate=get_train_UserUpdate()
    test_UserUpdate=get_test_UserUpdate()
    new_train_UserUpdate=feature_check(train_UserUpdate,test_UserUpdate)
    new_train_UserUpdate.to_csv('D:/kesci/data/part_data/train_UserUpdate.csv',index=None)
    test_UserUpdate.to_csv('D:/kesci/data/part_data/test_UserUpdate.csv',index=None)
    

    
    

    
