# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 20:42:59 2018

@author: CHANDU
"""
import dill
dill.load_session('ford.pkl')

import os
import numpy as np
import pandas as pd
from scipy.stats import skew

os.chdir('C:\\Users\\Chandu\\Desktop\\IMAR DATA\\ford')
ford_train= pd.read_csv('fordTrain.CSV')
ford_test=pd.read_csv('fordTest.csv')

ford_train.columns

#['TrialID', 'ObsNum', 'IsAlert', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6',
#       'P7', 'P8', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10',
#       'E11', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
#       'V11'

summary= ford_train.describe()
ford_train.info()
ford_test.info()

 # ---------------------------------------analysis for continous-----------------------
def analysis_continous(x):
    a = x.value_counts()
    b = x.dtypes
    c= sum(x.isnull())
    d= skew(x)
    e= x.min()
    
    return a,b,c,d,e
    
def fn_test(x):
        if(skew(x)>1 and x.min()>=0):
            log= np.log1p(x)
            
        elif(skew(x)>1 and x.min()<=-1):
            b= 50+x
            log= np.log(b)
            
        else:
            
            log= x
            
    
        return log  
    
def fn_neg(x):
    if(skew(x)<-1 and x.min()>=0):
        a= pow(x,2)
        log=np.log1p(a)
    elif(skew(x)<-1 and x.min()<=-1):
        a=pow(x,2)
        log= np.log(a)
    else:
        log=x
        
    return log
    

#----------------------------------------X train VALUES--------------------------------------

 analysis_continous(ford_train['P1'])
 #positive skew so transform
 ford_train['P1']= fn_test(ford_train['P1'])
 skew(ford_train['P1'])
#-----------------------------------------------------------# 
 analysis_continous(ford_train['P2'])
 
 analysis_continous(ford_train['P3'])
 
 analysis_continous(ford_train['P4'])
#---------------------------p5------------------------------------------------------ 
 analysis_continous(ford_train['P5'])
 ford_train['q']= fn_test(ford_train['P5'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 
 
 ford_train['P5']= ford_train['q']
 skew(ford_train['P5'])
 
 ford_train['q']=0
 #------------------------------------------------------------
 
 analysis_continous(ford_train['P6'])
 ford_train['q']= fn_test(ford_train['P6'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 
 ford_train['P6']= ford_train['q']
 skew(ford_train['P6'])
 
 ford_train['q']=0
 
 #---------------------------------------------------------
 analysis_continous(ford_train['P7'])
 ford_train['q']= np.sqrt(ford_train['P7'])
 skew(ford_train['q'])
 
 ford_train['P7']= ford_train['q']
 skew(ford_train['P7'])
 
 ford_train['q']=0
 
 #------------------------------------drop P8 all zeros
 analysis_continous(ford_train['P8'])
 ford_train.drop('P8',axis=1,inplace= True)
 analysis_continous(ford_test['P8'])
 ford_test.drop('P8',axis=1,inplace= True)
 
 #-------------------------------------------------------------------
 analysis_continous(ford_train['E1'])
 
 analysis_continous(ford_train['E2'])
 
 analysis_continous(ford_train['E3'])
 
 analysis_continous(ford_train['E4'])
 ford_train['q']= fn_test(ford_train['P7'])
 skew(ford_train['q'])
 
 ford_train['E4']= ford_train['q']
 skew(ford_train['E4'])
 
 ford_train['q']=0
 
 #-----------------------
 analysis_continous(ford_train['E5'])
 
 analysis_continous(ford_train['E6'])
 
 analysis_continous(ford_train['E7'])
 
 analysis_continous(ford_train['E8'])
 
 analysis_continous(ford_train['E9'])
 
 analysis_continous(ford_train['E10'])
 #---------------------------------------
 #skew cannot be reduced gradually
 analysis_continous(ford_train['E11'])
 ford_train['q']= fn_test(ford_train['E11'])
 skew(ford_train['q'])
 
 ford_train['E11']= ford_train['q']
 skew(ford_train['E11'])
 
 ford_train['q']=0
 
 #--------------------------
 analysis_continous(ford_train['V1'])
 
 analysis_continous(ford_train['V2'])
 
 analysis_continous(ford_train['V3'])
 
 analysis_continous(ford_train['V4'])
 ford_train['q']= fn_test(ford_train['V4'])
 skew(ford_train['q'])
 analysis_continous(ford_train['q'])
 ford_train['q']= fn_test(ford_train['q'])
 skew(ford_train['q'])
 
 ford_train['V4']= ford_train['q']
 skew(ford_train['V4'])
 
 ford_train['q']=0
 
 #--------------------------
 analysis_continous(ford_train['V5'])
 
 analysis_continous(ford_train['V6'])
 
 #--------------------------------drop v7
 analysis_continous(ford_train['V7'])
 ford_train.drop('V7', axis=1,inplace= True)
 ford_test.drop('V7', axis=1,inplace= True)
 
 analysis_continous(ford_train['V8'])
 
 
 #-------------------------------drop v9------------------------ 
 analysis_continous(ford_train['V9'])
 ford_train.drop('V9', axis=1,inplace= True)
 ford_test.drop('V9', axis=1,inplace= True)
 
 analysis_continous(ford_train['V10'])
 
 analysis_continous(ford_train['V11'])
 ford_train['q']= fn_test(ford_train['V11'])
 skew(ford_train['q'])
 
 ford_train['V11']= ford_train['q']
 skew(ford_train['V11'])
 
 ford_train['q']=0
 
 #----------------------------------ford test transformation------------------------------
 analysis_continous(ford_test['P1'])
 #positive skew so transform
 ford_test['P1']= fn_test(ford_test['P1'])
 skew(ford_test['P1'])
#-----------------------------------------------------------# 
 analysis_continous(ford_test['P2'])
 
 analysis_continous(ford_test['P3'])
 
 analysis_continous(ford_test['P4'])
#---------------------------p5------------------------------------------------------ 
 analysis_continous(ford_test['P5'])
 ford_test['q']= fn_test(ford_test['P5'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 
 
 ford_test['P5']= ford_test['q']
 skew(ford_test['P5'])
 
 ford_test['q']=0
 #------------------------------------------------------------
 
 analysis_continous(ford_test['P6'])
 ford_test['q']= fn_test(ford_test['P6'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 
 ford_test['P6']= ford_test['q']
 skew(ford_test['P6'])
 
 ford_test['q']=0
 
 #---------------------------------------------------------
 analysis_continous(ford_test['P7'])
 ford_test['q']= np.sqrt(ford_test['P7'])
 skew(ford_test['q'])
 
 ford_test['P7']= ford_test['q']
 skew(ford_test['P7'])
 
 ford_test['q']=0
 
 #------------------------------------drop P8 all zeros


 #-------------------------------------------------------------------
 analysis_continous(ford_test['E1'])
 
 analysis_continous(ford_test['E2'])
 
 analysis_continous(ford_test['E3'])
 
 analysis_continous(ford_test['E4'])
 ford_test['q']= fn_test(ford_test['P7'])
 skew(ford_test['q'])
 
 ford_test['E4']= ford_test['q']
 skew(ford_test['E4'])
 
 ford_test['q']=0
 
 #-----------------------
 analysis_continous(ford_test['E5'])
 
 analysis_continous(ford_test['E6'])
 
 analysis_continous(ford_test['E7'])
 
 analysis_continous(ford_test['E8'])
 
 analysis_continous(ford_test['E9'])
 
 analysis_continous(ford_test['E10'])
 #---------------------------------------
 #skew cannot be reduced gradually
 analysis_continous(ford_test['E11'])
 ford_test['q']= fn_test(ford_test['E11'])
 skew(ford_test['q'])
 
 ford_test['E11']= ford_test['q']
 skew(ford_test['E11'])
 
 ford_test['q']=0
 
 #--------------------------
 analysis_continous(ford_test['V1'])
 
 analysis_continous(ford_test['V2'])
 
 analysis_continous(ford_test['V3'])
 
 analysis_continous(ford_test['V4'])
 ford_test['q']= fn_test(ford_test['V4'])
 skew(ford_test['q'])
 analysis_continous(ford_test['q'])
 ford_test['q']= fn_test(ford_test['q'])
 skew(ford_test['q'])
 
 ford_test['V4']= ford_test['q']
 skew(ford_test['V4'])
 
 ford_test['q']=0
 
 #--------------------------
 analysis_continous(ford_test['V5'])
 
 analysis_continous(ford_test['V6'])
 
 #--------------------------------drop v7

 
 analysis_continous(ford_test['V8'])
 
 
 #-------------------------------drop v9------------------------ 
 
 analysis_continous(ford_test['V10'])
 
 analysis_continous(ford_test['V11'])
 ford_test['q']= fn_test(ford_test['V11'])
 skew(ford_test['q'])
 
 ford_test['V11']= ford_test['q']
 skew(ford_test['V11'])
 
 ford_test['q']=0

#----------------------splitting train data into x_train y_train x_test y_test
 
 X= ford_train.drop('IsAlert',axis=1)
 Y= ford_train['IsAlert']
 
 from sklearn.model_selection import train_test_split
 x_train,x_test,y_train,y_test= train_test_split(X,Y, test_size=0.2)
 
 
#----------------------------decision tre--------------------
 from sklearn.tree import DecisionTreeClassifier
 dt= DecisionTreeClassifier()
 model_dt= dt.fit(x_train,y_train)
 predicted_dt_train= model_dt.predict(x_train)
 
confusion_matrix_dt_train= confusion_matrix(y_train,predicted_dt_train)
classification_report_dt_train= classification_report(y_train,predicted_dt_train)
accucracy_dt_train= accuracy_score(y_train,predicted_dt_train) 

predicted_dt_test= model_dt.predict(x_test)
confusion_matrix_dt_test= confusion_matrix(y_test,predicted_dt_test)
classification_report_dt_test= classification_report(y_test,predicted_dt_test)
accucracy_dt_test= accuracy_score(y_test,predicted_dt_test) 

#---------------------tree visulaization------------------------
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydot

features= list(X.columns[0:])
print(features)
data= StringIO()
export_graphviz(dt,out_file=data,feature_names=features,filled=True,rounded=False)
graph= pydot.graph_from_dot_data(data.getvalue())
Image(graph[0].create_png())
Image(graph[0].write_png('tree.png'))



 
 
 #----------------------------------------------------------------
from sklearn import svm
sv= svm.SVC(kernel='rbf')

model_sv= sv.fit(x_train,y_train)
predicted_sv= model_sv.predict(x_train)
confusion_x_train= confusion_matrix(y_train,predicted_sv)
accurcay_x_train_score= accuracy_score(y_train,predicted_sv)
print(accurcay_x_train_score)
 
#--------------------------------MLC 
#-----------------confusion matrix accucary------------------
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score


#---------------------------MLP CLASSIFIER---------------------------------
from sklearn.neural_network import MLPClassifier
MLC= MLPClassifier(activation='relu',solver= 'adam',
                  max_iter=200, 
                  hidden_layer_sizes=(100,100,100))
model_mlc= MLC.fit(x_train,y_train)
predicted_mlc= MLC.predict(x_train)
mlc_confusing_matrix_y_train= confusion_matrix(y_train,predicted_mlc)
accuracy_y_train= accuracy_score(y_train,predicted_mlc)
print(accuracy_y_train)
 #----fit to   x_test,y_test-------for MLP CLASSIFIER-------------------------------------
predicted_mlc_test= model_mlc.predict(x_test)
mlc_confusing_matrix_y_test=confusion_matrix(y_test,predicted_mlc_test) 
accuracy_y_test= accuracy_score(y_test,predicted_mlc_test)
print(accuracy_y_test)



#------------------------------------test-----------------------------------------

ford_test['IsAlert']=0
ford_test['IsAlert']= model_mlc.predict(ford_test)
 
 ford_test.drop('q',axis=1,inplace=True)
 
 ford_test.to_csv('output.csv')
 
 
 import dill
filename = 'ford.pkl'
dill.dump_session(filename)
 
 
 
 
 
 
 