import pandas as pd
import numpy as np
import re
import pickle

bankdata = pd.read_csv('D:\\data_science _data _set\\Loan_prediction\\final_source_data\\train.csv')

#Train data
#bankdata6.shape #o/p:(104999, 27)
trndt1 = bankdata.copy()
#trndt1.shape  #(104999, 27)
trndt1.head(10)

#EDA for Train DATA
trndt1.shape  #(104999, 27)
#Found no duplicates in the dataframe
any(trndt1.duplicated())                  #o/p:False
#Found duplicate records in all the columns of the dataframe
#any(trndt1['Name'].duplicated())          #o/p:True
#any(trndt1['ChgOffPrinGr'].duplicated())  #o/p:True
#Number of missing/null values in the data
trndt1.isnull().sum()
#Observed Null values in the following columns with their count:
#Bank=111,BankState=112,RevLineCr=14,ChgOffDate=76722,DisbursementDate=156,MIS_Status=615

#Removing null values from Train data
trndt1 = trndt1.dropna(axis=0,subset = ['Name','City','State','Bank','BankState','RevLineCr','DisbursementDate','MIS_Status'])
#trndt1.isnull().sum()    #o/p:only in ChgOffDate column:76035 na values are present now
#After removing na values
#trndt1.shape   #o/p:(104145, 27)
trndt1.head(1)

#Dropping unnecessary columns from Train Dataset
#Dropping 'Unnamed: 0' column(as it is unnecessary for our analysis),and 'ChgOffDate' column as it has more than 75% missing data,from the Dataframe
trndt1.drop(columns = ['Unnamed: 0','ChgOffDate'], axis=1, inplace=True)
trndt1.shape #o/p:(104145, 25)

#Train Data
#converting MIS_Status column to numeric
#trndt1.MIS_Status.value_counts()
#o/p:P I F 76962
#CHGOFF    27183
#Name: MIS_Status, dtype: int64
trndt1['MIS_Status'] = trndt1['MIS_Status'].apply(lambda x: 1 if x == 'CHGOFF' else 0)
#trndt1.MIS_Status.value_counts()
#o/p:1 76962
#0     27183
#Name: MIS_Status, dtype: int64
trndt1.head(1)

#Train Data
trndt1['MIS_Status'].value_counts(normalize=True)
trndt1['MIS_Status'].value_counts()
#1    0.261011
#Name: MIS_Status, dtype: float64

#From Train Data removing the dollar sign and converting the following columns to numeric 
trndt1[['DisbursementGross','BalanceGross','ChgOffPrinGr','GrAppv','SBA_Appv']] = trndt1[['DisbursementGross','BalanceGross','ChgOffPrinGr','GrAppv','SBA_Appv']].replace('[\$,]','',regex=True).astype(float)
#trndt1.head(1)
trndt1[['DisbursementGross','BalanceGross','ChgOffPrinGr','GrAppv','SBA_Appv']] = trndt1[['DisbursementGross','BalanceGross','ChgOffPrinGr','GrAppv','SBA_Appv']].astype(int)
trndt1.head(1)


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
#trndt1[['ApprovalDate','Approval_NewDate']].head(1)
#trndt1['Approval_NewDate'].dtypes        #o/p:dtype('O')
#converting the newly created column from object(string) to int and checking it's datatype
trndt1['Approval_Date'] = trndt1['Approval_Date'].astype(int)
#trndt1['Approval_Date'].dtypes
trndt1['Approval_Month'] = trndt1['Approval_Month'].astype(int)
#trndt1['Approval_Month'].dtypes
trndt1['Approval_Year'] = trndt1['Approval_Year'].astype(int)
#trndt1['Approval_Year'].dtypes
trndt1['Approval_NewDate'] = trndt1['Approval_NewDate'].astype(int)
trndt1['Approval_NewDate'].dtypes          #o/p:dtype('int64')

#In Train Data,converting the DisbursementDate column to numeric
trndt1['DisbursementDate'] = trndt1['DisbursementDate'].replace(0, np.NaN)
trndt1['Disbursement_Date'],trndt1['Disbursement_Month'],trndt1['Disbursement_Year']=trndt1['DisbursementDate'].str.split('-',2).str
#trndt1[['DisbursementDate','Disbursement_Date','Disbursement_Month','Disbursement_Year']].head(1)
#converting the month names to month numbers in Disbursement Date column
month_numbers = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07','Aug': '08','Sep': '09','Oct': '10','Nov': '11','Dec': '12'}
for k, v in month_numbers.items(): 
    trndt1['Disbursement_Month'] = trndt1['Disbursement_Month'].replace(k, v)
#trndt1['Disbursement_Month'].head(1)
#concatenating all the 3 columns which are in numeric form as a single New column and checking it's datatype
trndt1['Disbursement_NewDate'] = trndt1[['Disbursement_Date','Disbursement_Month','Disbursement_Year']].apply(lambda x: ''.join(x),axis=1)
#trndt1[['DisbursementDate','Disbursement_NewDate']].head(1)
#o/p:DisbursementDate	Disbursement_Date1
#0	    31-Jul-98	         310798
trndt1['Disbursement_NewDate'].dtypes        #o/p:dtype('O')
#converting the newly created column from object(string) to int and checking it's datatype
trndt1['Disbursement_Date'] = trndt1['Disbursement_Date'].astype(int)
#trndt1['Disbursement_Date'].dtypes 
trndt1['Disbursement_Month'] = trndt1['Disbursement_Month'].astype(int)
#trndt1['Disbursement_Month'].dtypes
trndt1['Disbursement_Year'] = trndt1['Disbursement_Year'].astype(int)
#trnd1['Disbursement_Year'].dtypes  
trndt1['Disbursement_NewDate'] = trndt1['Disbursement_NewDate'].astype(int)
trndt1['Disbursement_NewDate'].dtypes         #o/p:dtype('int64')



#Train Data
trndt1.shape    #o/p:(104145, 33)
#trndt1.dtypes
#trndt1.info()  #o/p:dtypes: int64(23), object(9)

#listendata woe and iv article coding
#After removing null values and removing a column('chgoffdate') as it has more than 75% missing data
#After converting $ columns,ApprovalDate and DisbursementDate columns to numeric 
##checking for woe and iv 
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
##Rules related to Information Value
#Information Value	          Variable Predictiveness                columns
#Less than 0.02	              Not useful for prediction              BalanceGross,Approval_NewDate,Approval_Date,Disbursement_Month,NewExist,FranchiseCode,Approval_Month,Disbursement_Date(8)
#0.02 to 0.1	                Weak predictive Power                  CreateJob,Disbursement_NewDate,Zip,NoEmp,State,DisbursementGross,CCSC(7)
#0.1 to 0.3	                  Medium predictive Power                LowDoc,GrAppv,RetainedJob,SBA_Appv,RevLineCr(5)
#0.3 to 0.5	                  Strong predictive Power                BankState,UrbanRural,Approval_Year,ApprovalFY,Disbursement_Year(5)
#>0.5	                        Suspicious Predictive Power            City,Bank,ApprovalDate,DisbursementDate,Name,Term,ChgOffPrinGr(7)

##predictors
#not considering:notuseful(8)+weak(7) = Total = 15
#considering:medium(5)+strong(5) = Total = 10
#if we consider:suspicious(7) = Total = 10+7 = 17
#or else(medium+strong=5+5=10)
iv_18.sort_values(by='IV_18',ascending=False)

IvWoe_18.sort_values(by='WoE_18',ascending=False)

IvWoe_18.sort_values(by='IV_18',ascending=False)

woeiv_18  #o/p:120074 rows × 9 columns

#Train Data
trndt2 = trndt1.copy()
trndt2.shape     #o/p:(104145, 33)

##Based on Information Value Interpretation choosen the following columns for model building
#Information Value            Variable Predictiveness                Columns
#0.1 to 0.3	                  Medium predictive Power                LowDoc,GrAppv,RetainedJob,SBA_Appv,RevLineCr(5)
#0.3 to 0.5	                  Strong predictive Power                BankState,UrbanRural,Approval_Year,ApprovalFY,Disbursement_Year(5)

#In Train Data,converting the categorical columns 'BankState','RevLineCr','LowDoc' into numeric by using label encoder
trndt1.BankState.value_counts()
trndt1.BankState.value_counts().count() #o/p:52

#Train Data
from sklearn.preprocessing import LabelEncoder
le_0 = LabelEncoder()
# Encode labels in column 'BankState'
## instantiate an encoder - here we use labelencoder()
trndt1["BankState"] = trndt1["BankState"].astype(str)
trndt1['BankState_code']= le_0.fit_transform(trndt1['BankState'])
trndt1[['BankState_code','BankState']]
trndt1.head(1)

#Train Data
trndt1["RevLineCr"].value_counts()
#trndt1["RevLineCr"].value_counts().count() #o/p:7

#Train Data
#from sklearn.preprocessing import LabelEncoder
le_1 = LabelEncoder()
# Encode labels in column 'RevLineCr'
## instantiate an encoder - here we use labelencoder()
trndt1["RevLineCr"] = trndt1["RevLineCr"].astype(str)
trndt1['RevLineCr_code']= le_1.fit_transform(trndt1['RevLineCr'])
trndt1.head(1)

#Train Data
trndt1["LowDoc"].value_counts()
#trndt1["LowDoc"].value_counts().count()  #o/p:4

#Train Data
#from sklearn.preprocessing import LabelEncoder
le_2 = LabelEncoder()
# Encode labels in column 'LowDoc'
## instantiate an encoder - here we use labelencoder()
trndt1["LowDoc"] = trndt1["LowDoc"].astype(str)
trndt1['LowDoc_code']= le_2.fit_transform(trndt1['LowDoc'])
trndt1.head(1)


#Train Data
#Scaling the numerical columns before building the model
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
trndt1[['GrAppv', 'SBA_Appv']] = scaler.fit_transform(trndt1[['GrAppv', 'SBA_Appv']])
trndt1.head(2)

#Train Data
#After scaling the numerical variable columns save as copy
trndt4 = trndt1.copy()
trndt4.shape      #o/p:(104145, 36)
#trndt4.info()     
#The scaled 2-numerical variables have a data type:float 
#o/p:dtypes: float64(2), int64(25), object(9)

#Train Data
#selecting the columns based on Information Value
#medium predictors-LowDoc,GrAppv,RetainedJob,SBA_Appv,RevLineCr
#strong predictors-BankState,UrbanRural,Approval_Year,ApprovalFY,Disbursement_Year
trndt1.head(1)

#Train Data
#Remove the columns as they are not required for analysis 
trndt1.drop(trndt1.columns[[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 22, 25, 26, 28, 29, 30, 32]], axis = 1, inplace = True)

#Train Data
#After retaining the medium and strong predictive columns based on information value
trndt1.head(1)

#Train Data
#trndt1.shape     #o/p:(104145, 11)
#trndt1.info()    #o/p:dtypes: float64(2), int64(9)
trndt5 = trndt1.copy()
trndt5.shape      #o/p:(104145, 11)
#trndt5.info()    #o/p:dtypes: float64(2), int64(9)

#trndt4.shape     #(104145, 36)
#testdt4.shape    # (44900, 35)
trndt4.info()    #o/p:dtypes: float64(2), int64(25), object(9)


# Separate input features (X) and target variable (y)
X_train_Features = trndt1.drop('MIS_Status', axis=1)
y_train_label = trndt1.MIS_Status

# import SMOTE module from imblearn library
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2) 
X_train_Features_sm_1, y_train_label_sm_1 = sm.fit_sample(X_train_Features, y_train_label.ravel()) 

X_train_Features_sm_1.columns

#Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,confusion_matrix,roc_auc_score,roc_curve,auc
# Create DecisionTree model
clf_DT = DecisionTreeClassifier()
clf_DT.fit(X_train_Features_sm_1, y_train_label_sm_1)
# Train model and make predictions
pred_y_1 = clf_DT.predict(X_train_Features_sm_1)
confusion_matrix(y_train_label_sm_1,pred_y_1)


print("classification_report_DT:\n", classification_report(y_train_label_sm_1,pred_y_1))


print("AUC&ROC_DT:", roc_auc_score(y_train_label_sm_1,pred_y_1 ))
#o/p:AUC&ROC_DT: 0.8110519204972385

# Saving model to disk

pickle.dump(clf_DT, open('model.pkl','wb'))






