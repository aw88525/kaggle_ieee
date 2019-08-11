# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 22:00:52 2019

@author: wsy88
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics

print("start!")
train_transaction = pd.read_csv(r'F:\Users\vaskalab\kaggle\input\train_transaction.csv')
#test_transaction = pd.read_csv('/kaggle/input/test_transaction.csv')
train_identity = pd.read_csv(r'F:\Users\vaskalab\kaggle\input\train_identity.csv')
#test_identity = pd.read_csv('/kaggle/input/train_identity.csv')
print("finish loading the files!")
train_transaction_full = train_transaction.merge(train_identity, how = 'left', left_on = 'TransactionID', right_on = 'TransactionID')
#test_trainsaction = test_transaction.merge(test_identity, how = 'left', left_on = 'TransactionID', right_on = 'TransactionID')
fraud_set = train_transaction_full[train_transaction_full['isFraud'] == 1]
nonfraud_set = train_transaction_full[train_transaction_full['isFraud'] == 0]


usecolumns = []
len_columns = len(train_transaction_full.columns)
for i in range(len_columns):
    naportion = train_transaction_full.iloc[:, i].isna().sum() / len(train_transaction_full)
    if(naportion <= 0.95):
        usecolumns.append(train_transaction_full.columns[i])
        
train_transaction_use = train_transaction_full.loc[:, usecolumns]

column_names_cat = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
       'addr1', 'addr2', 'P_emaildomain','R_emaildomain', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18',
       'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32','id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType',
       'DeviceInfo']
#column_names_cat = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
#       'addr1', 'addr2', 'P_emaildomain', 'M4', 'M6']
column_names_notinterested = ['isFraud', 'TransactionID', 'TransactionDT']
column_names_noncat = [name for name in train_transaction_use.columns.tolist() if name not in column_names_cat and name not in column_names_notinterested]

for column in column_names_cat:
    cat = nonfraud_set.loc[:, column].value_counts(dropna = False).nlargest(n=20).index
    if(len(cat) >5):
        newcat = cat[:5]
    else:
        newcat = cat
    print(newcat)
    yesorno = [ value not in cat for value in train_transaction_use.loc[:,column]]
    train_transaction_use.loc[yesorno,column] = 'other'
    
    
from sklearn.preprocessing import Imputer
#train_transaction_use.replace('',np.nan)
fill_NaN = Imputer(missing_values=np.nan, strategy='median', axis = 1)


train_transaction_use_noncat = train_transaction_use.loc[:,column_names_noncat]

train_transaction_use_noncat_imputed = pd.DataFrame(fill_NaN.fit_transform(train_transaction_use_noncat))
train_transaction_use_noncat_imputed.columns = train_transaction_use_noncat.columns


for column in column_names_noncat:
    train_transaction_use_noncat_imputed[column] = train_transaction_use_noncat_imputed[column]
    
train_new = preprocessing.normalize(train_transaction_use_noncat_imputed.values)
train_transaction_use_noncat_imputed2 = pd.DataFrame(train_new)

train_transaction_use_noncat_imputed2.columns = train_transaction_use_noncat_imputed.columns
train_transaction_use_noncat_imputed = train_transaction_use_noncat_imputed2


train_transaction_use_cat = train_transaction_use.loc[:, column_names_cat]
train_transaction_use_catnew = pd.get_dummies(train_transaction_use_cat)


X_train_full = pd.concat([train_transaction_use_catnew, train_transaction_use_noncat_imputed], axis=1, sort=False)
y_train_full = train_transaction_use.loc[:,'isFraud']

X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=1)



clf = RandomForestClassifier(n_estimators=100, max_depth=20,
                             random_state=0)
clf.fit(X_train, y_train)  
y_pred_train = clf.predict(X_train)

confusion_matrix(y_train, y_pred_train)

y_pred_proba = clf.predict_proba(X_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


y_pred_test = clf.predict(X_test)
confusion_matrix(y_test, y_pred_test)