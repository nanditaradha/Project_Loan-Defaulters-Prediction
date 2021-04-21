import pickle
from flask import Flask,jsonify,request,render_template,make_response
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
le_0 = LabelEncoder()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index_new.html')


@app.route('/predict', methods=['POST'])    
def predict():
    request_file = request.files['data_file']
    print(1)
    if not request_file:
        return "No file"  
           
    df_loan=pd.read_csv(request_file)
    
    
    df_loan["RevLineCr"] = df_loan["RevLineCr"].astype(str)
    df_loan['RevLineCr_code']= le_0.fit_transform(df_loan['RevLineCr'])
    
    df_loan[['GrAppv','SBA_Appv']] = df_loan[['GrAppv','SBA_Appv']].replace('[\$,]','',regex=True).astype(float)
    df_loan[['GrAppv','SBA_Appv']] = df_loan[['GrAppv','SBA_Appv']].astype(int)
    
    scaler = preprocessing.MinMaxScaler()
    df_loan[['GrAppv', 'SBA_Appv']] = scaler.fit_transform(df_loan[['GrAppv', 'SBA_Appv']])
    
    df_loan['Approval_Date'],df_loan['Approval_Month'],df_loan['Approval_Year']=df_loan['ApprovalDate'].str.split('-',2).str
    df_loan['Approval_Year'] = df_loan['Approval_Year'].astype(int)
    
    print(df_loan['DisbursementDate'].isnull().value_counts())
    df_loan['Disbursement_Date'],df_loan['Disbursement_Month'],df_loan['Disbursement_Year']=df_loan['DisbursementDate'].str.split('-',2).str
    
    df_loan['Disbursement_Year'] = df_loan['Disbursement_Year'].fillna(0)
    print(df_loan['Disbursement_Year'].isnull().value_counts())
    
    
    df_loan['Disbursement_Year'] = df_loan['Disbursement_Year'].astype(int)
    
    
    df_loan["BankState"] = df_loan["BankState"].astype(str)
    df_loan['BankState_code']= le_0.fit_transform(df_loan['BankState'])
    
    df_loan['LowDoc_code']= le_0.fit_transform(df_loan['LowDoc'])
    
    df_loan=df_loan[['ApprovalFY', 'RetainedJob', 'UrbanRural', 'GrAppv', 'SBA_Appv',
       'Approval_Year', 'Disbursement_Year', 'BankState_code',
       'RevLineCr_code', 'LowDoc_code']]
    
    with open('model.pkl', 'rb') as f:
        dt = pickle.load(f)
    print("loading saved artifacts...done")
    value=dt.predict(df_loan)
    print(value)
    value_df=pd.DataFrame(value,columns=['Prediction1'])
    value_df['Prediction1']=value_df['Prediction1'].apply(lambda x: 'CHGOFF' if x==1 else 'P I F')
    value_df.index+=1
    result=value_df.to_csv(sep=',',index_label='Id')
    response = make_response(result)
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return response
              
if __name__ == '__main__':
    app.run()

