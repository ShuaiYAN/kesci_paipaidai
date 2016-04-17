
# coding: utf-8



import pandas as pd 

predict_1=pd.read_csv('.../xgb_1_test_predict.csv',encoding="gb18030")
predict_2=pd.read_csv('.../xgb_2_test_predict.csv',encoding="gb18030")
predict_3=pd.read_csv('.../xgb_3_test_predict.csv',encoding="gb18030")

predict_target_1=predict_1['score']
predict_target_2=predict_2['score']
predict_target_3=predict_3['score']

result=predict_target_1*0.33+predict_target_2*0.34+predict_target_3*0.33

#sum=0
#for i in range(len(result)):
#    if result[i]>0.5:
#        sum+=1
#print(sum)
#print(len(result))


test_result=pd.DataFrame(columns=['Idx','score'])
test_result.Idx=predict_1['Idx']
test_result.score=result
test_result.to_csv("I:/all_data/final_result.csv",index=None)





