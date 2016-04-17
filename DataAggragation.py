#encoding=utf-8

import pandas as pd
from Data import load_file
from sklearn.preprocessing import Imputer


dir='D:/kesci/data/part_data'
test_master_numeric='/test_master_numeric.csv'
test_master_category='/test_master_category.csv'
test_UserUpdate='/test_UserUpdate.csv'
test_LogInfo='/test_LogInfo.csv'
train_master_numeric='/train_master_numeric.csv'
train_master_category='/train_master_category.csv'
train_UserUpdate='/train_UserUpdate.csv'
train_LogInfo='/train_LogInfo.csv'

test_master_1=load_file(dir,test_master_numeric)
test_master_2=load_file(dir,test_master_category)
train_master_1=load_file(dir,train_master_numeric)
train_master_2=load_file(dir,train_master_category)

test_UserUpdate_3=load_file(dir,test_UserUpdate)
test_LogInfo_4=load_file(dir,test_LogInfo)
train_UserUpdate_3=load_file(dir,train_UserUpdate)
train_LogInfo_4=load_file(dir,train_LogInfo)


train=pd.concat([train_master_1,train_master_2],axis=1)
test=pd.concat([test_master_1,test_master_2],axis=1)
print(train.shape)
print(test.shape)

train=pd.merge(train,train_UserUpdate_3,how='left',on='Idx')
train=pd.merge(train,train_LogInfo_4,how='left',on='Idx')
print(train.shape)

test=pd.merge(test,test_UserUpdate_3,how='left',on='Idx')
test=pd.merge(test,test_LogInfo_4,how='left',on='Idx')
print(test.shape)

train=Imputer().fit_transform(train)
test=Imputer().fit_transform(test)
print(train.shape)
print(test.shape)
train=pd.DataFrame(train)
test=pd.DataFrame(test)

train.to_csv('D:/kesci/data/part_data/train.csv',index=None)
test.to_csv('D:/kesci/data/part_data/test.csv',index=None)
