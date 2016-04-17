
# coding: utf-8

import random
from sklearn.cross_validation import train_test_split
import pandas as pd
import xgboost as xgb

train_master=pd.read_csv('.../train_master.csv')
train_predict=pd.read_csv('.../xgb_1_train_predict.csv')


#得到分类错误的数据集和同等数量分类正确的构成新的数据集并重新分类
train_target=train_master[['Idx','target']]
train_predict.score[train_predict['score']>0.5]=1
train_predict.score[train_predict['score']<0.5]=0
ture=[]
false=[]
for i in range(len(train_target)):
    if train_target.target[i]==train_predict.score[i]:
        ture.append(train_target.Idx[i])
    else:
        false.append(train_target.Idx[i])

ture_sample=random.sample(ture,5618)
ture_sample.extend(false)
sample=pd.Series(ture_sample,name='Idx')

train_x=pd.read_csv('.../train.csv')
test_x=pd.read_csv('.../test.csv')

temp=train_x['0'].isin(sample)
train=train_x[temp]
target=train_target[temp].target

random_seed=1225

#get the y 
label=target
#delete idx in x

train=train.drop('0',axis=1)

#the full_train and test
#train_idx=train_x['0']
#train_x=train_x.drop("0",axis=1)
test_idx=test_x['0']
test=test_x.drop('0',axis=1)

X,val_X,y,val_y=train_test_split(train,label,test_size=0.2,random_state=random_seed)

#xgboost start here
#dtrain_test = xgb.DMatrix(train_x)
dtest=xgb.DMatrix(test)

dval = xgb.DMatrix(val_X,label=val_y)
dtrain = xgb.DMatrix(X, label=y)
params={
    'booster':'gbtree',
    'objective':'binary:logistic',
    'early_stopping_rounds':50,
    'eval_metric':'auc',
    'gamma':0.1,
    'max_depth':3,
    'lambda':10,
    'subsample':0.5,
    'colsample_bytree':0.4,
    'min_child_weight':1,
    'eta':0.01,
    'seed':random_seed,
    'nthread':3
    }

watchlist=[(dval,'val'),(dtrain,'train')]
model=xgb.train(params,dtrain,num_boost_round=4000,evals=watchlist)
model.save_model("xgb_2.model")


#test_train=model.predict(dtrain_test,ntree_limit=model.best_ntree_limit)
#test_train_result=pd.DataFrame(columns=['Idx','score'])
#test_train_result.Idx=train_idx
#test_train_result.score=test_train
#test_train_result.to_csv("xgb_2_train_predict.csv",index=None,encoding='utf-8')

test_y=model.predict(dtest,ntree_limit=model.best_ntree_limit)
test_result=pd.DataFrame(columns=['Idx','score'])
test_result.Idx=test_idx
test_result.score=test_y
test_result.to_csv("xgb_2_test_predict.csv",index=None,encoding='utf-8')





