#!/usr/bin/env python  

from sklearn.cross_validation import train_test_split
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import Imputer


random_seed=1225
train_x_csv=".../train.csv"
train_master_csv =".../train_master.csv"
test_x_csv =".../test.csv"


#get the y 
train_master=pd.read_csv(train_master_csv)
label=train_master['target']

#read the train and test x
train_x = pd.read_csv(train_x_csv)
test_x= pd.read_csv(test_x_csv)

#delete idx in x
#train=train_x.drop("0",axis=1)
test=test_x.drop('0',axis=1)
#train_idx=train_x['0']
test_idx=test_x['0']

#Imp=Imputer(strategy='most_frequent')
#train_x=Imp.fit_transform(train_x)
#test_x=Imp.fit_transform(test_x)
#train_x=pd.DataFrame(train_x)
#test_x=pd.DataFrame(test_x)


X,val_X,y,val_y=train_test_split(train,label,test_size=0.2,random_state=random_seed)


#xgboost start here
#dtrain_test = xgb.DMatrix(train)
dtest=xgb.DMatrix(test)

dval = xgb.DMatrix(val_X,label=val_y)
dtrain = xgb.DMatrix(X, label=y)
params={
    'booster':'gbtree',
    'objective':'binary:logistic',
    'early_stopping_rounds':50,
    'eval_metric':'auc',
    'gamma':0.1,
    'max_depth':5,
    'lambda':10,
    'subsample':0.5,
    'colsample_bytree':0.4,
    'min_child_weight':1,
    'eta':0.01,
    'seed':random_seed,
    'nthread':3
    }

watchlist=[(dval,'val'),(dtrain,'train')]
model=xgb.train(params,dtrain,num_boost_round=2500,evals=watchlist)
model.save_model("xgb_1.model")


#test_train=model.predict(dtrain_test,ntree_limit=model.best_ntree_limit)
#test_train_result=pd.DataFrame(columns=['Idx','score'])
#test_train_result.Idx=train_idx
#test_train_result.score=test_train
#test_train_result.to_csv("xgb_1_train_predict.csv",index=None,encoding='utf-8')

test_y=model.predict(dtest,ntree_limit=model.best_ntree_limit)
test_result=pd.DataFrame(columns=['Idx','score'])
test_result.Idx=test_idx
test_result.score=test_y
test_result.to_csv("xgb_1_test_predict.csv",index=None,encoding='utf-8')




              
