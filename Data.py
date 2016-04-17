#encoding=utf-8


import pandas as pd

#训练集路径
dir='D:/kesci/PPD-Second-Round-Data'
#test_target_1='/初赛测试集/daily_test.csv'
#test_target_2='/初赛测试集/final_test.csv'
first_test_Master_data='/初赛测试集/Kesci_Master_9w_gbk_2.csv'
first_test_LogInfo_data='/初赛测试集/LogInfo_9w_2.csv'
first_test_UserUpdateInfo_data='/初赛测试集/UserUpdate_Info_9w_2.csv'
first_train_Master_data='/初赛训练集/PPD_Training_Master_GBK_3_1_Training_Set.csv'
first_train_LogInfo_data='/初赛训练集/PPD_LogInfo_3_1_Training_Set.csv'
first_train_UserUpdateInfo_data='/初赛训练集/PPD_Userupdate_Info_3_1_Training_Set.csv'
second_train_Master_data='/复赛新增数据/Kesci_Master_9w_gbk_3_2.csv'
second_train_LogInfo_data='/复赛新增数据/LogInfo_9w_3_2.csv'
second_train_UserUpdateInfo_data='/复赛新增数据/Userupdate_Info_9w_3_2.csv'

#测试集路径
test_master_data='/复赛测试集/Kesci_Master_9w_gbk_1_test_set.csv'
test_logInfo_9w_1='/复赛测试集/LogInfo_9w_1.csv'
test_Userupdate_Info_9w_1='/复赛测试集/Userupdate_Info_9w_1.csv'


#数据读取
def load_file(dir,filename):
       file=pd.read_csv(dir+filename,sep=",",header=0,encoding="gb18030")
       return file

#合成初赛和复赛的master，并生成新的train_master
first_train_master=load_file(dir,first_train_Master_data)
first_test_master=load_file(dir,first_test_Master_data)
second_train_master=load_file(dir,second_train_Master_data)
train_master=pd.concat([first_train_master,first_test_master,second_train_master])

#合成初赛和复赛的LogInfo,并生成新的train_logInfo
first_train_LogInfo=load_file(dir,first_train_LogInfo_data)
first_test_LogInfo=load_file(dir,first_test_LogInfo_data)
second_train_LogInfo=load_file(dir,second_train_LogInfo_data)
train_LogInfo=pd.concat([first_train_LogInfo,first_test_LogInfo,second_train_LogInfo])

#合成初赛和复赛的UserUpdateInfo,并生成新的train_UserUpdateInfo
first_train_UserUpdateInfo=load_file(dir,first_train_UserUpdateInfo_data)
first_test_UserUpdateInfo=load_file(dir,first_test_UserUpdateInfo_data)
second_train_UserUpdateInfo=load_file(dir,second_train_UserUpdateInfo_data)
train_UserUpdateInfo=pd.concat([first_train_UserUpdateInfo,first_test_UserUpdateInfo,second_train_UserUpdateInfo])

#输出
train_master.to_csv('D:/kesci/data/train/train_master.csv',index=None)
train_LogInfo.to_csv('D:/kesci/data/train/train_LogInfo.csv',index=None)
train_UserUpdateInfo.to_csv('D:/kesci/data/train/train_UserUpdateInfo.csv',index=None)
