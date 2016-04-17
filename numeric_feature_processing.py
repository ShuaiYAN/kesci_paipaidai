#encoding='utf-8'


import pandas as pd
from Data import load_file
import FeatureCombine as FC

#master数据路径
dir='D:/kesci'
feature_Info_path='/PPD-Second-Round-Data/魔镜杯风控数据字段类型补充说明.csv'
train_master_path='/data/train/train_master.csv'
test_master_path='/PPD-Second-Round-Data/复赛测试集/Kesci_Master_9w_gbk_1_test_set.csv'

#获取数值型特征并去除批准日期
def get_numeric_feature(featureInfo):
    numeric=featureInfo[featureInfo["Index"]=="Numerical"]
    numeric_list=numeric["Idx"]
    numeric_list=numeric_list.drop([226],axis=0)
    print(numeric_list)
    return numeric_list

#得到通过相关系数筛选后单变量特征列表
def get_single_feature_list(train_master,numeric_list):
    train_master_numeric_data=train_master[numeric_list]
    target=train_master_numeric_data['target']
    corr_list=[target.corr(train_master_numeric_data[i]) for i in numeric_list.drop(225,axis=0)]
    num_list1=[i for i in numeric_list]
    num_list1=pd.Series(num_list1)
    corr_list=pd.Series(corr_list)
    num_feature_corr=pd.concat([num_list1,abs(corr_list)],axis=1)
    num_feature_corr.columns=['Idx','corr']
    single_feature=num_feature_corr[num_feature_corr['corr']>0.03]
    single_feature_list=single_feature['Idx']
    return single_feature_list


#生成plus特征
def get_feature_plus(feature_plus_list,master_data,new_data):
    feature_plus_list=pd.DataFrame(feature_plus_list)
    for i in range(len(feature_plus_list)):
        f1=feature_plus_list.iloc[i][0]
        f2=feature_plus_list.iloc[i][1]
        temp=master_data[f1]+master_data[f2]
        new_data=pd.concat([new_data,temp],axis=1)
    print(new_data.shape)
    return new_data 

#生成minus特征
def get_feature_minus(feature_minus_list,master_data,new_data):
    feature_minus_list=pd.DataFrame(feature_minus_list)
    for i in range(len(feature_minus_list)):
        f1=feature_minus_list.iloc[i][0]
        f2=feature_minus_list.iloc[i][1]
        temp=master_data[f1]-master_data[f2]
        new_data=pd.concat([new_data,temp],axis=1)
    print(new_data.shape)
    return new_data

#生成Mulitiple特征
def get_feature_mul(feature_mul_list,master_data,new_data):
    feature_mul_list=pd.DataFrame(feature_mul_list)
    for i in range(len(feature_mul_list)):
        f1=feature_mul_list.iloc[i][0]
        f2=feature_mul_list.iloc[i][1]
        temp=master_data[f1]*master_data[f2]
        new_data=pd.concat([new_data,temp],axis=1)
    print(new_data.shape)
    return new_data

#生成Div特征
def get_feature_div(feature_div_list,master_data,new_data):
    feature_div_list=pd.DataFrame(feature_div_list)
    for i in range(len(feature_div_list)):
        f1=feature_div_list.iloc[i][0]
        f2=feature_div_list.iloc[i][1]
        temp=master_data[f1]/master_data[f2]
        new_data=pd.concat([new_data,temp],axis=1)
    print(new_data.shape)
    return new_data

def get_train_master_numeric():
    train_data=train_master[single_feature_list]
    train_data=get_feature_plus(feature_plus_list,train_master,train_data)
    train_data=get_feature_minus(feature_minus_list,train_master,train_data)
    train_data=get_feature_mul(feature_mul_list,train_master,train_data)
    train_data=get_feature_div(feature_div_list,train_master,train_data)
    train_idx=train_master['Idx']
    train_data=pd.concat([train_idx,train_data],axis=1)
    print(train_data.shape)
    return train_data

def get_test_master_numeric():
    test_data=test_master[single_feature_list]
    test_data=get_feature_plus(feature_plus_list,test_master,test_data)
    test_data=get_feature_minus(feature_minus_list,test_master,test_data)
    test_data=get_feature_mul(feature_mul_list,test_master,test_data)
    test_data=get_feature_div(feature_div_list,test_master,test_data)
    test_idx=test_master['Idx']
    test_data=pd.concat([test_idx,test_data],axis=1)
    print(test_data.shape)
    return test_data
    


if __name__=='__main__':
    #读入数据
    test_master=load_file(dir,test_master_path)
    featureInfo=load_file(dir,feature_Info_path)
    train_master=load_file(dir,train_master_path)
#得到数值型列表
    numeric_list=get_numeric_feature(featureInfo)
#得到单变量特征列表
    single_feature_list=get_single_feature_list(train_master,numeric_list)
#数值型进行加减乘除组合
    train_master_numeric_data=train_master[numeric_list]
    eps=0.08
    feature_plus_list=FC.feature_Plus(train_master_numeric_data,numeric_list,eps)
    feature_mul_list=FC.feature_Mul(train_master_numeric_data,numeric_list,eps)
    feature_div_list=FC.feature_Div(train_master_numeric_data,numeric_list,eps)
    feature_minus_list=FC.feature_Minus(train_master_numeric_data,numeric_list,eps)
#得到train_master的numeric特征
    train_master_numeric=get_train_master_numeric()
    test_master_numeric=get_test_master_numeric()
    train_master_numeric.to_csv('D:/kesci/data/part_data/train_master_numeric.csv',index=None)
    test_master_numeric.to_csv('D:/kesci/data/part_data/test_master_numeric.csv',index=None)
    
    




