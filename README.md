# Project_Loan-Defaulters-Prediction

## Description & Background

The Dataset provided for the project have the all the similar features as that to U.S.Small Business Administration(SBA).

## About SBA

The U.S.SBA was founded in 1953 on the principle of promoting and assisting small enterprises in the U.S. credit market (SBA Overview and History,US Small Business Administration (2015)).Small businesses have been a primary source of job creation in the United States; therefore,fostering small business formation and growth has social benefits by creating job opportunities and reducing unemployment.One way SBA assists these small business enterprises is through a loan guarantee program which is designed to encourage banks to grant loans to small businesses.SBA acts much like an insurance provider to reduce the risk for a bank by taking on some of the risk through guaranteeing a portion of the loan.In the case that a loan goes into default,SBA then covers the amount they guaranteed.There have been many success stories of start-ups receiving SBA loan guarantees such as FedEx and Apple Computer.However,there have also been stories of small businesses and/or start-ups that have defaulted on their SBA-guaranteed loans.The rate of default on these loans has been a source of controversy for decades.[reference](https://www.tandfonline.com/doi/full/10.1080/10691898.2018.1434342)

## Project Overview

Using the partially provided dataset along with Machine Learning Algorithms we can predict the risky customers.Additionally,using the results from the predictive modelling we can improve the potential rate of Loan Approval.

### Business Objective

To Predict whether the customer will Default or Not.

## Table Of Contents

* Introduction
* Data Description
* Model Details
* Model Evaluation
* Model Prediction
* Installation
* Model Deployment
* Directory Tree
* Technologies Used

## Introduction

With an intention to help the approached customers who wish to start Small Business,SBA preliminarily checks upon the Bank provided customer information and checks for prior Number of Defaulters based on the Banking terms 'CHGOFF'(Charge-Off/Defaulted) and 'PIF'(Paid-In-Full) and updates the information to the Bank and encourages them to take the necessary steps before sanctioning the prescribed loan amount to be paid as partial or in full mode which is requested by the customer.

As the customers are categorized into two classes Defaulters and Non-Defaulters,it relates to a Binary classification problem and the future model which will be built by using the partial dataset provided to us has to Identify and Predict the Maximum Number of Defaulters inorder to save the Bank from facing future risks and financial crisis.In a classification problem,we have to predict discrete values based on a given set of independent variable(s).

## Data Description

We are provided with two Datasets one is Train and the other is Test.
The Train Dataset contains 26 input variables with 150000 observations.
The Test Dataset contains 25 variables with 45000 observations.

* Train Dataset will be used for training the model,i.e. our model will learn from this Dataset.It contains all the independent variables and the target variable.
* Test Dataset contains all the independent variables,but not the target variable.We will apply the model to predict the target variable for the test data.

Set of features:
 <p align="center">
   </p>
   Each Row Represents A client's Information
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>City</th>
      <th>State</th>
      <th>Zip</th>
      <th>Bank</th>
      <th>BankState</th>
      <th>CCSC</th>
      <th>ApprovalDate</th>
      <th>ApprovalFy</th>
      <th>Term</th>
      <th>NoEmp</th>
      <th>NewExist</th>
      <th>CreateJob</th>
      <th>RetainedJob</th>
      <th>FranchiseCode</th>
      <th>UrbanRural</th>
      <th>RevLineCr</th>
      <th>LowDoc</th>
      <th>ChgOffDate</th>
      <th>DisbursementDate</th>
      <th>DisbursementGross</th>
      <th>SBA_Gross</th>
      <th>MIS_Status</th>
      <th>ChgOffPrinGr</th>
      <th>GrAppv</th>
      <th>SBA_Appv</th>    
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PATTERSON FARM</td>
      <td>CARNESVILLE</td>
      <td>GA</td>
      <td>30521</td>
      <td>SOUTH STATE BANK</td>
      <td>GA</td>
      <td>112310</td>
      <td>2-Mar-98</td>
      <td>1998</td>
      <td>180</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>N</td>
      <td>N</td>
      <td>Nan</td>
      <td>31-Jul-98</td>
      <td>$765,000.00</td>
      <td>$0.00</td>
      <td>PIF</td>
      <td>$0.00</td>
      <td>$765,000.00</td>
      <td>$573,750.00</td>     
    </tr>
  </tbody>
</table>
</div>

## Model Details

* Based on the Interpretation of Weight Of Evidence(WOE) and Information values(IV),I was able to chose a set of features that can be used for model building.[reference for WOE & IV](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)
* As the Train Dataset is imbalanced in class with respect to Dependent Target variable(MIS_Status),SMOTE with minority oversampling technique was employed to handle the class imbalance in the said Target variable.[reference for Class Imbalance](https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18)
* I started this step by comparing the baseline model for a few algorithms which are Random Forest classification,KNN classification,Naive Bayes classification and Decision Tree classification.The baseline model then evaluated using precision, recall and roc_auc metrics for 10 fold cross validation.And Finally Decision Tree Classifier was chosen for Model Building.

## Model Evaluation

1. Random Forest : 89%
2. KNN : 88%
3. Naive Bayes : 49%
4. Decision Tree : 89.2%

## Model Predictions

When the trained model was fitted on Test Dataset it was able to identify and predict the number of Defaulters.

* Number of Defaulters(CHG-OFF) : 11746
* Number of Non-Defaulters(PIF) : 33154

## Installation
The Code is written in Python 3.7. If you don't have Python installed you can find it [here](https://www.python.org/downloads/).If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
```bash
pip install -r requirements.txt
```
## Model Deployment

The built model after training and evaluation was deployed using Flask, as an app on Heroku Platform. 

Create a new repository on [GitHub](https://github.com/) and upload the project.

Follow the instructions given in [Heroku Documentation](https://devcenter.heroku.com/articles/getting-started-with-python) to deploy a web app.

## Directory Tree 
```
├── templates 
│   └── index.html
├── app.py
├── credit-card-default.csv
├── credit_default_prediction.py
├── model.pkl
├── Procfile
├── README.md
└── requirements.txt
```

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://numpy.org/images/logos/numpy.svg" width=100>](https://numpy.org)    [<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/450px-Pandas_logo.svg.png" width=150>](https://pandas.pydata.org)    [<img target="_blank" src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width=150>](https://scikit-learn.org/stable)

[<img target="_blank" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png" width=150>](https://flask.palletsprojects.com/en/1.1.x/) [<img target="_blank" src="https://number1.co.za/wp-content/uploads/2017/10/gunicorn_logo-300x85.png" width=200>](https://gunicorn.org) [<img target="_blank" src="https://matplotlib.org/_static/logo2_compressed.svg" width=170>](https://matplotlib.org)      [<img target="_blank" src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" width=150>](https://seaborn.pydata.org)

[<img target="_blank" src="https://jupyter.org/assets/nav_logo.svg" width=150>](https://jupyter.org)



