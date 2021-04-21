import pandas as pd
import numpy as np
import re
import pickle

bankdata = pd.read_csv('D:\\Loan_Prediction_defaulters\\deployment_app_smote_decision_tree\\train.csv')

#1.EDA-Train Dataset
#1.Data Pre-Processing
bankdata.shape              #o/p:(104999, 27)
trndt1 = bankdata.copy()
trndt1.shape               #o/p:(104999, 27)
trndt1.head(10)            #o/p:[10 rows x 27 columns]

#Found no duplicates in the dataframe
any(trndt1.duplicated())                    #o/p:False
#Found duplicate records in all the columns of the dataframe
#any(trndt1['Name'].duplicated())          #o/p:True
#any(trndt1['ChgOffPrinGr'].duplicated())  #o/p:True

#checking for null values if any
trndt1.isnull().sum()
#Observed Null values in the following columns with their count:
#Name=6,City=1,State=1,Bank=111,BankState=112,RevLineCr=14,ChgOffDate=76722,DisbursementDate=156,MIS_Status=615

#2.Data Cleaning
#a.Removing null values from Train data
trndt1 = trndt1.dropna(axis=0,subset = ['Name','City','State','Bank','BankState','RevLineCr','DisbursementDate','MIS_Status'])
trndt1.isnull().sum()    #o/p:only in ChgOffDate column:76035 na values are present now

#After removing na values
trndt1.shape             #o/p:(104145, 27)
trndt1.head(1)           #o/p:[1 rows x 27 columns]

#b.Dropping unnecessary columns from Train Dataset
#Dropping 'Unnamed: 0' column(as it is unnecessary for our analysis),and 'ChgOffDate' column as it has more than 75% missing data,from the Dataframe
trndt1.drop(columns = ['Unnamed: 0','ChgOffDate'], axis=1, inplace=True)
trndt1.shape            #o/p:(104145, 25)

#3.Conversions of DataTypes in columns
#a.converting MIS_Status column to numeric
trndt1.MIS_Status.value_counts()
#o/p:P I F     76962
#    CHGOFF    27183
#Name: MIS_Status, dtype: int64

#Converting 'CHGOFF' and  'PIF' to '1' and '0' respectively
trndt1['MIS_Status'] = trndt1['MIS_Status'].apply(lambda x: 1 if x == 'CHGOFF' else 0)
trndt1.MIS_Status.value_counts()
#o/p:0 76962
#    1 27183
#Name: MIS_Status, dtype: int64
trndt1.head(1)

#Train Data
trndt1['MIS_Status'].value_counts(normalize=True)
#trndt1['MIS_Status'].value_counts()
#o/p:0    0.738989
#    1    0.261011
#Name: MIS_Status, dtype: float64

#b.From Train Data removing the dollar sign and converting the following columns to numeric 
trndt1[['DisbursementGross','BalanceGross','ChgOffPrinGr','GrAppv','SBA_Appv']] = trndt1[['DisbursementGross','BalanceGross','ChgOffPrinGr','GrAppv','SBA_Appv']].replace('[\$,]','',regex=True).astype(float)
#trndt1.head(1)
trndt1[['DisbursementGross','BalanceGross','ChgOffPrinGr','GrAppv','SBA_Appv']] = trndt1[['DisbursementGross','BalanceGross','ChgOffPrinGr','GrAppv','SBA_Appv']].astype(int)
trndt1.head(1)              #o/p:[1 rows x 25 columns]

#4.Derived Metrics
#Deriving Date,Month and Year for ApprovalDate,DisbursementDate columns
#In Train Data,converting the ApprovalDate column to numeric
trndt1['Approval_Date'],trndt1['Approval_Month'],trndt1['Approval_Year']=trndt1['ApprovalDate'].str.split('-',2).str
#trndt1[['ApprovalDate','Approval_Date','Approval_Month','Approval_Year']].head(1)
#converting the month names to month numbers in Approval Date column
month_numbers = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07','Aug': '08','Sep': '09','Oct': '10','Nov': '11','Dec': '12'}
for k, v in month_numbers.items(): 
    trndt1['Approval_Month'] = trndt1['Approval_Month'].replace(k, v)
#trndt1['Approval_Month'].head(1)
#concatenating all the 3 columns which are in numeric form as a single New column and checking it's datatype
trndt1['Approval_NewDate'] = trndt1[['Approval_Date','Approval_Month','Approval_Year']].apply(lambda x: ''.join(x),axis=1)
trndt1[['ApprovalDate','Approval_NewDate']].head(1)
trndt1['Approval_NewDate'].dtypes          #o/p:dtype('O')
#converting the newly created column from object(string) to int and checking it's datatype
trndt1['Approval_Date'] = trndt1['Approval_Date'].astype(int)
trndt1['Approval_Date'].dtypes             #o/p:dtype('int32')
trndt1['Approval_Month'] = trndt1['Approval_Month'].astype(int)
trndt1['Approval_Month'].dtypes            #o/p:dtype('int32')
trndt1['Approval_Year'] = trndt1['Approval_Year'].astype(int)
trndt1['Approval_Year'].dtypes             #o/p:dtype('int32')
trndt1['Approval_NewDate'] = trndt1['Approval_NewDate'].astype(int)
trndt1['Approval_NewDate'].dtypes          #o/p:dtype('int32')

#In Train Data,converting the DisbursementDate column to numeric
trndt1['Disbursement_Date'],trndt1['Disbursement_Month'],trndt1['Disbursement_Year']=trndt1['DisbursementDate'].str.split('-',2).str
#trndt1[['DisbursementDate','Disbursement_Date','Disbursement_Month','Disbursement_Year']].head(1)
#converting the month names to month numbers in Disbursement Date column
month_numbers = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07','Aug': '08','Sep': '09','Oct': '10','Nov': '11','Dec': '12'}
for k, v in month_numbers.items(): 
    trndt1['Disbursement_Month'] = trndt1['Disbursement_Month'].replace(k, v)
#trndt1['Disbursement_Month'].head(1)
#concatenating all the 3 columns which are in numeric form as a single New column and checking it's datatype
trndt1['Disbursement_NewDate'] = trndt1[['Disbursement_Date','Disbursement_Month','Disbursement_Year']].apply(lambda x: ''.join(x),axis=1)
trndt1[['DisbursementDate','Disbursement_NewDate']].head(1)
#o/p:DisbursementDate	Disbursement_Date1
#0	    31-Jul-98	         310798
trndt1['Disbursement_NewDate'].dtypes        #o/p:dtype('O')
#converting the newly created column from object(string) to int and checking it's datatype
trndt1['Disbursement_Date'] = trndt1['Disbursement_Date'].astype(int)
trndt1['Disbursement_Date'].dtypes           #o/p:dtype('int32')
trndt1['Disbursement_Month'] = trndt1['Disbursement_Month'].astype(int)
trndt1['Disbursement_Month'].dtypes          #o/p:dtype('int32')
trndt1['Disbursement_Year'] = trndt1['Disbursement_Year'].astype(int)
trndt1['Disbursement_Year'].dtypes           #o/p:dtype('int32')
trndt1['Disbursement_NewDate'] = trndt1['Disbursement_NewDate'].astype(int)
trndt1['Disbursement_NewDate'].dtypes        #o/p:dtype('int32')

#Train Data
trndt1.shape     #o/p:(104145, 33)
#trndt1.dtypes
trndt1.info()    #o/p:dtypes:int32(13), int64(11), object(9)

#Train Data
trndt2 = trndt1.copy()
trndt2.shape     #o/p:(104145, 33)

#5.Converting columns into labels in Train Dataset before Building the Model
#Instantiate an encoder - here we use labelencoder()
from sklearn.preprocessing import LabelEncoder
#a.Encode labels in column 'BankState'
trndt1.BankState.value_counts()
trndt1.BankState.value_counts().count() #o/p:52

le_0 = LabelEncoder()
trndt1["BankState"] = trndt1["BankState"].astype(str)
trndt1['BankState_code']= le_0.fit_transform(trndt1['BankState'])
trndt1.head(1)      #o/p:[1 rows x 34 columns]

#b.Encode labels in column 'RevLineCr'
trndt1["RevLineCr"].value_counts()
trndt1["RevLineCr"].value_counts().count() #o/p:7

le_1 = LabelEncoder()
trndt1["RevLineCr"] = trndt1["RevLineCr"].astype(str)
trndt1['RevLineCr_code']= le_1.fit_transform(trndt1['RevLineCr'])
trndt1.head(1)     #o/p:[1 rows x 35 columns]

#c.Encode labels in column 'LowDoc'
trndt1["LowDoc"].value_counts()
trndt1["LowDoc"].value_counts().count()  #o/p:4

le_2 = LabelEncoder()
trndt1["LowDoc"] = trndt1["LowDoc"].astype(str)
trndt1['LowDoc_code']= le_2.fit_transform(trndt1['LowDoc'])
trndt1.head(1)   #o/p:[1 rows x 36 columns]

#Train Data
#trndt1.shape    #o/p:(104145, 36)
#trndt1.dtypes
#trndt1.info()   #o/p:dtypes: int32(16), int64(11), object(9)
trndt3 = trndt1.copy()
trndt3.shape    #o/p:(104145, 36)
trndt3.info()   #o/p:dtypes: int32(16), int64(11), object(9)

#6.Scaling the numerical columns in Train Data before building the model in Train Data
from sklearn import preprocessing
scaler_Train = preprocessing.MinMaxScaler()
trndt1[['GrAppv', 'SBA_Appv']] = scaler_Train.fit_transform(trndt1[['GrAppv', 'SBA_Appv']])
trndt1.head(1)

#Train Data
#After scaling the numerical variable columns save it as copy
trndt4 = trndt1.copy()
trndt4.shape      #o/p:(104145, 36)
trndt4.info()     
#The scaled 2-numerical variables have a data type:float 
#o/p:dtypes: float64(2), int32(14), int64(11), object(9)

#Followed with reference to listendata woe and iv article coding
#Calculating Weight Of Evidence(WOE) and Information Value(IV) for the all the Features w.r.to the Target Variable in Train Data.
#After removing null values and removing a column('chgoffdate') as it has more than
# 75% missing data
#After converting $ columns,ApprovalDate and DisbursementDate columns to numeric
#After converting the columns Bank,BankState,RevLinCr and LowDcoc into labels(by using label encoder())
#After scaling the numerical columns which are 'GrAppv', 'SBA_Appv' in Train Data before building the model
##checking for woe and iv 
##7.checking for woe and iv 
def iv18_woe18(data, target, bins=10, show_woe=False):
    
    #Empty Dataframe
    newDF64,newDF65,newDF66,newDF67,woeivDF18 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    #Extract Column Names
    cols = data.columns
    
    #Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE_18'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['IV_18'] = d['WoE_18'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("Information value of " + ivars + " is " + str(round(d['IV_18'].sum(),6)))
        print("Weight Of Evidence of " + ivars + " is " + str(round(d['WoE_18'].sum(),6)))
        temp_9 =pd.DataFrame({"Variable" : [ivars], "IV_18" : [d['IV_18'].sum()]}, columns = ["Variable", "IV_18"])
        tempWOE_18 =pd.DataFrame({"variable" : [ivars], "WoE_18" : [d['WoE_18'].sum()]}, columns = ["variable", "WoE_18"])
        tempIV_18 =pd.DataFrame({"variable" : [ivars], "IV_18" : [d['IV_18'].sum()]}, columns = ["variable", "IV_18"])        
        tempIVWOE_18 =pd.DataFrame({"variable" : [ivars], "IV_18" : [d['IV_18'].sum()], "WoE_18" : [d['WoE_18'].sum()]}, columns = ["variable", "IV_18", "WoE_18"])

        newDF64=pd.concat([newDF64,temp_9], axis=0)
        newDF65=pd.concat([newDF65,tempWOE_18], axis=0)
        newDF66=pd.concat([newDF66,tempIV_18], axis=0)
        newDF67=pd.concat([newDF67,tempIVWOE_18], axis=0)
        woeivDF18=pd.concat([woeivDF18,d], axis=0)
        ##newDF3=pd.concat([newDF3,tempWOE], axis=0)
        #woeivDF1=pd.concat([woeivDF1,d], axis=0)

        #Show WOE Table
        if show_woe == True:
            print(d)
    return newDF64,newDF65,newDF66,newDF67,woeivDF18

final_18,woe_18,iv_18,IvWoe_18,woeiv_18 = iv18_woe18(data = trndt1, target = 'MIS_Status', bins=10, show_woe = True)
final_18
#print(woe_18)
#print(iv_18)
#print(IvWoe_18)
#print(woeiv_18)

type(final_18)         #o/p:pandas.core.frame.DataFrame

#Weight of Evidence(WOE):
#11-columns with +ve WOE:Name,City,DisbursementDate,ApprovalDate,ChgOffPrinGr,RetainedJob,RevLineCr,CCSC,BalanceGross,CreateJob,Disbursement_Month.
#21-columns with _ve WOE:Approval_NewDate,Approval_Date,Approval_Month,Zip,Disbursement_NewDate,Disbursement_Date,DisbursementGross,NoEmp,FranschiseCode,
#UrbanRural,ApprovalFY,Term,GrAppv,SBA_Appv,Approval_Year,NewExist,LowDoc,Disbursement_Year,State,BankState,Bank.
woe_18.sort_values(by='WoE_18',ascending=False)

#INFORMATION VALUE(IV):  
##Rules related to Information        Value
#Information Value	          Variable Predictiveness                      columns
#Less than 0.02	              Not useful for prediction            BalanceGross,Approval_NewDate,Approval_Date,Disbursement_Month,NewExist,FranchiseCode,Approval_Month,Disbursement_Date(8)
#0.02 to 0.1	              Weak predictive Power                CreateJob,Disbursement_NewDate,Zip,NoEmp,State,DisbursementGross,CCSC(7)
#0.1 to 0.3	                  Medium predictive Power              LowDoc,GrAppv,RetainedJob,SBA_Appv,RevLineCr(5)
#0.3 to 0.5	                  Strong predictive Power              BankState,UrbanRural,Approval_Year,ApprovalFY,Disbursement_Year(5)
#>0.5	                      Suspicious Predictive Power          City,Bank,ApprovalDate,DisbursementDate,Name,Term,ChgOffPrinGr(7)

##predictors
#not considering:notuseful(8)+weak(7) = Total = 15
#considering:medium(5)+strong(5) = Total = 10
#if we consider:suspicious(7) = Total = 10+7 = 17
#or else(medium+strong=5+5=10)
iv_18.sort_values(by='IV_18',ascending=False)

IvWoe_18.sort_values(by='WoE_18',ascending=False)

IvWoe_18.sort_values(by='IV_18',ascending=False)

woeiv_18                  #o/p:120094 rows × 9 columns

#Train Data
#10.Remove unnesseccary columns which are not required for analysis 
trndt1.drop(trndt1.columns[[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 22, 25, 26, 28, 29, 30, 32]], axis = 1, inplace = True)
trndt1.shape             #o/p:(104145, 11)

##Based on Information Value of Interpretation choosen the following columns for model building
#Information Value            Variable Predictiveness                Columns
#0.1 to 0.3	                  Medium predictive Power                LowDoc,GrAppv,RetainedJob,SBA_Appv,RevLineCr(5)
#0.3 to 0.5	                  Strong predictive Power                BankState,UrbanRural,Approval_Year,ApprovalFY,Disbursement_Year(5)

#Train Data
#trndt1.shape     #o/p:(104145, 11)
trndt1.info()    #o/p:dtypes: float64(2), int32(5), int64(4)
#Save the selected variables by making it as a copy.
trndt5 = trndt1.copy()
trndt5.shape      #o/p:(104145, 11)
#trndt5.info()    #o/p:dtypes: float64(2), int32(5), int64(4)

#11.saving the cleaned trained data for further analysis
trndt1.to_csv(r'D:\Loan_Prediction_Defaulters\deployment_app_smote_decision_tree\CleanedTrain_Data.csv')

#12.Separating the Feature(X) and Label(Target-y)variable from the Train Data
X_train_Features = trndt1.drop('MIS_Status',axis=1)
y_train_label = trndt1.MIS_Status

X_train_Features.columns

#13.Applying SMOTE on Train Data
#As the Target variable in the Train Data is categorical and is imbalanced,so we apply smote for the Train Data.
#Using SMOTE Algorithm
print("Before OverSampling, counts of label '1': {}".format(sum(y_train_label == 1))) 
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train_label == 0))) 
#o/p:Before OverSampling, counts of label '1': 27183
#Before OverSampling, counts of label '0': 76962
  
#import SMOTE module from imblearn library
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2) 
X_train_Features_sm_1, y_train_label_sm_1 = sm.fit_sample(X_train_Features, y_train_label.ravel()) 
#X_train_Features_sm_1, y_train_Label_sm_1 = sm.fit_sample(X_train_Features, y_train_Label) 

print('After OverSampling, the shape of train_X_Features_sm_1: {}'.format(X_train_Features_sm_1.shape)) 
print('After OverSampling, the shape of train_y_label_sm_1: {} \n'.format(y_train_label_sm_1.shape)) 
#o/p:After OverSampling, the shape of train_X_Features_sm_1: (153924, 10)
#After OverSampling, the shape of train_y_lable_sm_1: (153924,) 

print("After OverSampling, counts of label '1': {}".format(sum(y_train_label_sm_1 == 1))) 
print("After OverSampling, counts of label '0': {}".format(sum(y_train_label_sm_1 == 0)))
#o/p:After OverSampling, counts of label '1': 76962
#After OverSampling, counts of label '0': 76962 

X_train_Features_sm_1.columns
#o/p:Index(['ApprovalFY', 'RetainedJob', 'UrbanRural', 'GrAppv', 'SBA_Appv',
#       'Approval_Year', 'Disbursement_Year', 'BankState_code',
#       'RevLineCr_code', 'LowDoc_code'],
#      dtype='object')


#14.Decision Tree Classifier for Train Data after applying SMOTE
from sklearn.tree import DecisionTreeClassifier
#Create a Decision Tree classifier model
clf_DT = DecisionTreeClassifier()

#15.Fit on training data after applying Smote on Train Data
clf_DT.fit(X_train_Features_sm_1,y_train_label_sm_1)
#o/p:DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
#                       max_depth=one, max_features=None, max_leaf_nodes=None,
#                       min_impurity_decrease=0.0, min_impurity_split=None,
#                       min_samples_leaf=1, min_samples_split=2,
#                       min_weight_fraction_leaf=0.0, presort='deprecated',
#                       random_state=None, splitter='best')

#16.Calculating accuracy score for Train set after applying Smote on Train Data
#accuracy on Train set
train_sm_result_1 = clf_DT.score(X_train_Features_sm_1, y_train_label_sm_1)
print("Accuracy-Train-set-sm_1: %.2f%%" % (train_sm_result_1*100.0))
#o/p:Accuracy-Train-set-sm_1: 89.20%

#17.Actual class predictions after applying Smote on Train Data
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,roc_curve,auc,recall_score,f1_score,precision_score
pred_y_sm_1 = clf_DT.predict(X_train_Features_sm_1)
print("ACCURACY_Train-set-sm_dt1b:",accuracy_score(y_train_label_sm_1,pred_y_sm_1))
#o/p:ACCURACY_Train-set-sm_dt1b: 0.8920051453964294

#18.AUC-ROC Score after applying Smote on Train Data
print("AUC&ROC_Train-set-sm_dt1b:", roc_auc_score(y_train_label_sm_1,pred_y_sm_1))
#o/p:AUC&ROC_Train-set-sm_dt1b: 0.8920051453964294

#19.#ROC CURVE for Decision Tree Classifier after applying SMOTE on Train Data
import matplotlib.pyplot as plt
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import roc_curve
train_sm_roc_auc = roc_auc_score(y_train_label_sm_1, clf_DT.predict(X_train_Features_sm_1))
fpr_train_sm, tpr_train_sm, thresholds_train_sm = roc_curve(y_train_label_sm_1, clf_DT.predict_proba(X_train_Features_sm_1)[:,1])
plt.figure(figsize=(7,5))
plt.plot(fpr_train_sm, tpr_train_sm, label='Decision Tree Classifier-SM (area = %0.2f)' % train_sm_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic-Decision Tree Classifier-Train-SM')
plt.legend(loc="lower right")
#plt.savefig('DT_TRAIN_SM_ROC')
plt.show()

#20.Confusion Matrix After Applying SMOTE on Train Data
from sklearn.metrics import confusion_matrix
cm_sm = confusion_matrix(y_train_label_sm_1,pred_y_sm_1)
print(cm_sm)
#o/p:[[66894 10068]
#    [ 6555 70407]]

#21.Calculating Specificity After Applying SMOTE on Train Data
specificity_train_sm = cm_sm[1,1]/(cm_sm[1,0]+cm_sm[1,1])
print('Specificity-train-sm: ', specificity_train_sm)
#o/p:Specificity-train-sm :  0.9148280969829267

#22.Classification Report After Applying SMOTE on Train Data(train_sm_1)
print("classification_report_Train_sm_1:\n", classification_report(y_train_label_sm_1,pred_y_sm_1))
#o/p:classification_report_Train_sm_1:
#               precision    recall  f1-score   support

#           0       0.91      0.87      0.89     76962
#           1       0.87      0.91      0.89     76962

#    accuracy                           0.89    153924
#   macro avg       0.89      0.89      0.89    153924
#weighted avg       0.89      0.89      0.89    153924

#23.#10-Fold cross validation for Decision Tree Classifier after applying SMOTE on Train Data(dt1b)
from sklearn.model_selection import cross_val_score
scores_cv_train_sm = cross_val_score(clf_DT, X_train_Features_sm_1,y_train_label_sm_1, cv = 10)
print("Cross-validation scores for Train-sm-1: {}".format(scores_cv_train_sm))
print("Average cross-validation score for Train-sm-1: {:.2f}".format(scores_cv_train_sm.mean()))
#o/p:Cross-validation scores for Train-sm-1: [0.69682993 0.68520203 0.68834459 0.73460239 0.77293399 0.77780665
#0.77111486 0.7772869  0.77455821 0.78215956]
#Average cross-validation score for Train-sm-1: 0.75

#24.Decision Tree Classifier model results for Train Set(DT-sm-train(dt1b)) after applying SMOTE on Train Data
#ACCURACY:89%
#PRECISION:87%
#RECALL(SENSITIVITY):91%
#F1-SCORE:88%
#SPECIFICITY:91%
#AUC_ROC_SCORE:0.8920(89%)
#CROSS_VALIDATION MEAN SCORE:0.75(75%)

###############################################
###Test Dataset
#1.Loading the Test Data
testdt = pd.read_csv('D:\\Loan_Prediction_Defaulters\\deployment_app_smote_decision_tree\\test.csv')
testdt.shape        #o/p:(45000, 26)
testdt.head(10)
testdt.columns
testdt1 = testdt.copy()

#EDA for Test Data
#2.Data Pre=Processing
#Found no duplicate records in Test Dataframe
any(testdt1.duplicated())          #o/p:False

#Checking for Null/Missing values in Test Dataframe
testdt1.isnull().sum()
#Observed Null values in the following columns with their count:
#Bank=36,BankState=36,RevLineCr=14,ChgOffDate=32811,DisbursementDate=69

#3.Data Cleaning
#a.Removing null values from Test data
testdt1 = testdt1.dropna(axis=0,subset = ['Name','City','State','Bank','BankState','RevLineCr','DisbursementDate'])
testdt1.isnull().sum()    #o/p:only in ChgOffDate column:32748 na values are present now
testdt1.shape             #o/p:(44891, 26)
#After removing na values
testdt1.head(1)

#b.Dropping unnecessary columns from Test Dataset
#Dropping 'Unnamed: 0' column(as it is unnecessary for our analysis),and 'ChgOffDate' column as it has more than 75% missing data,from the Dataframe
testdt1.columns
testdt1.drop(columns = ['Unnamed: 0','ChgOffDate'], axis=1, inplace=True)
testdt1.shape                  #o/p:(44891, 24)
testdt1.head(1)

#4.DataType conversions
#Converting the DataType for a few columns in the Test Dataset
#From Test Data removing the dollar sign and converting the following columns to numeric 
testdt1[['DisbursementGross','BalanceGross','ChgOffPrinGr','GrAppv','SBA_Appv']] = testdt1[['DisbursementGross','BalanceGross','ChgOffPrinGr','GrAppv','SBA_Appv']].replace('[\$,]','',regex=True).astype(float)
#trndt1.head(1)
testdt1[['DisbursementGross','BalanceGross','ChgOffPrinGr','GrAppv','SBA_Appv']] = testdt1[['DisbursementGross','BalanceGross','ChgOffPrinGr','GrAppv','SBA_Appv']].astype(int)
testdt1.head(1)

#5.Derived Metrics
#Deriving DATE,MONTH and YEAR from ApprovalDate and DisbursementDate Columns
#In Test Data,converting the ApprovalDate column to numeric
testdt1['Approval_Date'],testdt1['Approval_Month'],testdt1['Approval_Year']=testdt1['ApprovalDate'].str.split('-',2).str
#test1[['ApprovalDate','Approval_Date','Approval_Month','Approval_Year']].head(1)
# converting the month names to month numbers in Approval Date column
month_numbers = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07','Aug': '08','Sep': '09','Oct': '10','Nov': '11','Dec': '12'}

for k, v in month_numbers.items(): 
    testdt1['Approval_Month'] = testdt1['Approval_Month'].replace(k, v)
testdt1['Approval_Month'].head(1)
#concatenating all the 3 columns which are in numeric form as a single New column and checking it's datatype
testdt1['Approval_NewDate'] = testdt1[['Approval_Date','Approval_Month','Approval_Year']].apply(lambda x: ''.join(x),axis=1)
testdt1[['ApprovalDate','Approval_NewDate']].head(1)
testdt1['Approval_NewDate'].dtypes        #o/p:dtype('O')
#converting the newly created column from object(string) to int and checking it's datatype
testdt1['Approval_Date'] = testdt1['Approval_Date'].astype(int)
testdt1['Approval_Date'].dtypes           #o/p:dtype('int32')
testdt1['Approval_Month'] = testdt1['Approval_Month'].astype(int)
testdt1['Approval_Month'].dtypes          #o/p:dtype('int32')
testdt1['Approval_Year'] = testdt1['Approval_Year'].astype(int)
testdt1['Approval_Year'].dtypes           #o/p:dtype('int32')
testdt1['Approval_NewDate'] = testdt1['Approval_NewDate'].astype(int)
testdt1['Approval_NewDate'].dtypes        #o/p:dtype('int64')

#In Test Data,converting the DisbursementDate column to numeric
testdt1['Disbursement_Date'],testdt1['Disbursement_Month'],testdt1['Disbursement_Year']=testdt1['DisbursementDate'].str.split('-',2).str
#test1[['DisbursementDate','Disbursement_Date','Disbursement_Month','Disbursement_Year']].head(1)
#converting the month names to month numbers in Disbursement Date column
month_numbers = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07','Aug': '08','Sep': '09','Oct': '10','Nov': '11','Dec': '12'}

for k, v in month_numbers.items(): 
    testdt1['Disbursement_Month'] = testdt1['Disbursement_Month'].replace(k, v)
#concatenating all the 3 columns which are in numeric form as a single New column and checking it's datatype
    testdt1['Disbursement_Date'].isnull().sum()   #0
    testdt1['Disbursement_Month'].isnull().sum() #61
    testdt1['Disbursement_Year'].isnull().sum()  #61
        
    testdt1['Disbursement_Year'] = testdt1['Disbursement_Year'].fillna(0)
    testdt1['Disbursement_Month'] = testdt1['Disbursement_Month'].fillna(0)

testdt1['Disbursement_NewDate'] = testdt1[['Disbursement_Date','Disbursement_Month','Disbursement_Year']].apply(lambda x: ''.join(x),axis=1)
testdt1[['Disbursement_Date','Disbursement_NewDate']].head(1)
#o/p:DisbursementDate	Disbursement_Date1
#0	    31	               311006
#testdt1['Disbursement_NewDate'].dtypes        #o/p:dtype('O')
#converting the newly created column from object(string) to int and checking it's datatype
testdt1['Disbursement_Date'] = testdt1['Disbursement_Date'].astype(int)
testdt1['Disbursement_Date'].dtypes            #o/p:dtype('32')
testdt1['Disbursement_Month'] = testdt1['Disbursement_Month'].fillna(0)
testdt1['Disbursement_Month'] = testdt1['Disbursement_Month'].astype(int)
testdt1['Disbursement_Month'].dtypes           #o/p:dtype('32')
testdt1['Disbursement_Year'] = testdt1['Disbursement_Year'].fillna(0)
testdt1['Disbursement_Year'] = testdt1['Disbursement_Year'].astype(int)
testdt1['Disbursement_Year'].dtypes            #o/p:dtype('32')
testdt1['Disbursement_NewDate'] = testdt1['Disbursement_NewDate'].astype(int)
testdt1['Disbursement_NewDate'].dtypes         #o/p:dtype('32')

#Test Data
testdt1.shape    #o/p:(44891, 32)
#testdt1.dtypes
#testdt1.info()  #o/p:dtypes: int64(24), object(9)
#Test Data
testdt2 = testdt1.copy()
testdt2.shape     #o/p:(44891, 32)

#6.In Test Data,converting the categorical columns 'BankState','RevLineCr','LowDoc'
# into numeric by using label encoder
testdt1.BankState.value_counts()
testdt1.BankState.value_counts().count()     #o/p:52

#Test Data
from sklearn.preprocessing import LabelEncoder
le_3 = LabelEncoder()
#Encode labels in column 'BankState'
#Instantiate an encoder - here we use labelencoder()
testdt1["BankState"] = testdt1["BankState"].astype(str)
testdt1['BankState_code']= le_3.fit_transform(testdt1['BankState'])
testdt1.head(1)            #o/p:[1 rows x 33 columns]

#Test Data
testdt1["RevLineCr"].value_counts()
testdt1["RevLineCr"].value_counts().count() #o/p:6
testdt1.head(1)            #o/p:[1 rows x 33 columns]

#Test Data
#from sklearn.preprocessing import LabelEncoder
le_4 = LabelEncoder()
#Encode labels in column 'RevLineCr'
#Instantiate an encoder - here we use labelencoder()
testdt1["RevLineCr"] = testdt1["RevLineCr"].astype(str)
testdt1['RevLineCr_code']= le_4.fit_transform(testdt1['RevLineCr'])
testdt1.head(1)            #o/p:[1 rows x 34 columns]
                          
#Test Data
testdt1["LowDoc"].value_counts()
testdt1["LowDoc"].value_counts().count()  #o/p:3

#Test Data
#from sklearn.preprocessing import LabelEncoder
le_5 = LabelEncoder()
#Encode labels in column 'LowDoc'
#Instantiate an encoder - here we use labelencoder()
testdt1["LowDoc"] = testdt1["LowDoc"].astype(str)
testdt1['LowDoc_code']= le_5.fit_transform(testdt1['LowDoc'])
testdt1.head(1)           #o/p:[1 rows x 35 columns]

#Test Data
#testdt1.shape      #o/p:(44891, 35)
#testdt1.dtypes
#testdt1.info()     #o/p:dtypes: int32(16), int64(10), object(9)
testdt3 = testdt1.copy()
testdt3.shape      #o/p:(44891, 35)
#testdt3.info()    #o/p:dtypes: int32(16), int64(10), object(9)

#7.Scaling the required numerical columns before fitting the model with Train Data 
#in order to find out the unknown variable in it 
from sklearn import preprocessing
scaler_Test = preprocessing.MinMaxScaler()
testdt1[['GrAppv', 'SBA_Appv']] = scaler_Test.fit_transform(testdt1[['GrAppv', 'SBA_Appv']])
testdt1.head(1)   #o/p:[1 rows x 35 columns]

#Test Data
#8.After scaling the numerical variable columns save them as a copy
testdt4 = testdt1.copy()
testdt4.shape      #o/p:(44891, 35)
#testdt4.info()     
#The scaled 2-numerical variables have a data type:float 
#o/p:dtypes: float64(2), int32(14), int64(10), object(9)
testdt1.head(1)   #o/p:[1 rows x 35 columns]

#Test Data
#9.Remove unnecessary columns as they are not required for analysis
#we are chosing those columns as per Training Data,as they were selected based on Information Interpretation Value
testdt1.drop(testdt1.columns[[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 21, 24, 25, 27, 28, 29, 31]], axis = 1, inplace = True)
testdt1.head(1)  #o/p:[1 rows x 10 columns]

#Test Data
testdt1.shape       #o/p:(44891, 10)
testdt1.info()      #o/p:dtypes: float64(2), int32(5), int64(3)

#10.Saving all the selected variables by making it as a copy
testdt5 = testdt1.copy()
testdt5.shape       #o/p:(44891, 10)
#testdt5.info()     #o/p:dtypes: float64(2), int32(5), int64(3)

#After retaining the medium and strong predictive columns based on information value
#saving the cleaned and formatted test data for further analysis.
testdt1.shape       #o/p:(44891, 10)

testdt1.to_csv(r'D:\Loan_Prediction_Defaulters\deployment_app_smote_decision_tree\CleanedTest_Data_.csv')

#11.To findout the unknown variable in Test Dataset using Decision Tree Classifier Algorithm
#X_train_Features_dt1a.shape   #o/p:(104145, 10)
#y_train_label_dt1a.shape      #o/p:(104145, )
#11.separting the Features and Target variable from Train Data
X_train_Features_dt1a = trndt1.drop('MIS_Status',axis=1)
y_train_label_dt1a = trndt1.MIS_Status

X_train_Features_dt1a.columns
#o/p:Index(['ApprovalFY', 'RetainedJob', 'UrbanRural', 'GrAppv', 'SBA_Appv',
#       'Approval_Year', 'Disbursement_Year', 'BankState_code',
#       'RevLineCr_code', 'LowDoc_code'],
#      dtype='object')

#12.Applying SMOTE on train set and fitting it on Test Dataset using Decision Tree Classifier Algorithm
#import SMOTE module from imblearn library
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2) 
X_train_Features_sm_1, y_train_label_sm_1 = sm.fit_sample(X_train_Features, y_train_label.ravel()) 

#13.Defining a Decision tree classifier
#from sklearn.tree import DecisionTreeClassifier
#Creating a Decision Tree Classifier
clf_dt1 = DecisionTreeClassifier()

#14.Fitting the Decision Tree Classifier on the Train Data
clf_dt1.fit(X_train_Features_dt1a,y_train_label_dt1a)
#o/p:DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
#                       max_depth=None, max_features=None, max_leaf_nodes=None,
#                       min_impurity_decrease=0.0, min_impurity_split=None,
#                       min_samples_leaf=1, min_samples_split=2,
#                       min_weight_fraction_leaf=0.0, presort='deprecated',
#                       random_state=None, splitter='best')

#15.To findout the unknown variable in Test Dataset using Decision Tree Classifier before applying SMOTE on Train Data
X_test_Features_dt2a = testdt1

#from sklearn.tree import DecisionTreeClassifier
#Create a DecisionTree classifier
#clf_dt1 = Decision TreeClassifier()
y_test_dt2a =clf_dt1.predict(X_test_Features_dt2a)  
y_test_dt2a
#o/p:array([1, 0, 0, ..., 0, 1, 0])

import pandas as pd
#converting the array into a pandas dataframe
dataset_y_dt2a = pd.DataFrame({'Default_y': y_test_dt2a[:]})
dataset_y_dt2a.head(5)
#o/p:Default_18a
#0	1
#1	0
#2	0
#3	0
#4	0

#16.Finding the unique counts for the Default_y variable in the Test Data
dataset_y_dt2a.Default_y.value_counts()
#o/p:0    31861
#    1    13030
#Name: Default_y, dtype: int64

##The Default_y variable is finally our Target variable

















