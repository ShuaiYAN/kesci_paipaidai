
# coding: utf-8


from sklearn.cross_validation import train_test_split
import xgboost as xgb
import pandas as pd 


train_master=pd.read_csv('train_master.csv')
train_target=train_master[['Idx','target']]

#two weak learning classifier
train_predict_1=pd.read_csv('xgb_1_train_predict.csv')
train_predict_2=pd.read_csv('xgb_2_train_predict.csv')


train_predict_1.score[train_predict_1['score']>0.5]=1
train_predict_1.score[train_predict_1['score']<0.5]=0

train_predict_2.score[train_predict_2['score']>0.5]=1
train_predict_2.score[train_predict_2['score']<0.5]=0

ture=[]
false=[]
for i in range(len(train_predict_1)):
    if train_predict_1.score[i]==train_predict_2.score[i]:
        ture.append(train_predict_1.Idx[i])
    else:
        false.append(train_predict_1.Idx[i])


train_x=pd.read_csv('train.csv')
test_x=pd.read_csv('test.csv')

temp=train_x['0'].isin(false)
train=train_x[temp]
target=train_target[temp].target


random_seed=1225

#get the y 
label=target
#delete idx in x
train=train.drop('0',axis=1)

#the full_train and test
train_idx=train_x['0']
train_x=train_x.drop("0",axis=1)
test_idx=test_x['0']
test=test_x.drop('0',axis=1)

X,val_X,y,val_y=train_test_split(train,label,test_size=0.2,random_state=random_seed)

#xgboost start here
dtrain_test = xgb.DMatrix(train_x)
dtest=xgb.DMatrix(test)

dval = xgb.DMatrix(val_X,label=val_y)
dtrain = xgb.DMatrix(X, label=y)
params={
    'booster':'gbtree',
    'objective':'binary:logistic',
    'early_stopping_rounds':50,
    'eval_metric':'auc',
    'gamma':0.1,
    'max_depth':7,
    'lambda':100,
    'subsample':0.5,
    'colsample_bytree':0.4,
    'min_child_weight':1,
    'eta':0.1,
    'seed':random_seed,
    'nthread':3
    }

watchlist=[(dval,'val'),(dtrain,'train')]
model=xgb.train(params,dtrain,num_boost_round=3000,evals=watchlist)
model.save_model("xgb_3.model")


test_train=model.predict(dtrain_test,ntree_limit=model.best_ntree_limit)
test_train_result=pd.DataFrame(columns=['Idx','score'])
test_train_result.Idx=train_idx
test_train_result.score=test_train
test_train_result.to_csv("xgb_3_train_predict.csv",index=None,encoding='utf-8')

test_y=model.predict(dtest,ntree_limit=model.best_ntree_limit)
test_result=pd.DataFrame(columns=['Idx','score'])
test_result.Idx=test_idx
test_result.score=test_y
test_result.to_csv("xgb_3_test_predict.csv",index=None,encoding='utf-8')





