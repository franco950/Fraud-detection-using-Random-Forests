import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV 
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score
from category_encoders import WOEEncoder
from win10toast import ToastNotifier
from sklearn.utils import resample
from treeinterpreter import treeinterpreter as ti
# def send_notification(message):
#     toaster = ToastNotifier()
#     toaster.show_toast("Notification", message, duration=20)

# # start=time.time()
# # stats=[]
# # csv_file_path='forestapp/fraudTrain.csv'
# # test_path='forestapp/fraudTest.csv'
# # df1=pd.read_csv(csv_file_path)
# # df2=pd.read_csv(test_path)

# # df = pd.concat([df1, df2], ignore_index=True)

# # # Save the combined DataFrame to a new CSV file
# # df.to_csv('combined_file.csv', index=False)
# # #dc=df.drop('Class',axis=1)
# # print(len(df))

# # print(len(df))

# # stats.append(df['is_fraud'].value_counts()) 
# # print(stats)

# # columns_to_drop = ['first', 'unix_time', 'dob', 'cc_num', 'zip', 'city','street', 'state', 'trans_num', 'trans_date_trans_time']
# # df=df.drop(columns_to_drop,axis=1)
# # df['merchant'] = df['merchant'].apply(lambda x : x.replace('fraud_',''))

# # df['gender'] = df['gender'].map({'F': 0, 'M': 1})

# # # applying WOE encoding
# # for col in ['job','merchant', 'category', 'lat', 'last']:
# #     df[col] = WOEEncoder().fit_transform(df[col],df['is_fraud'])

# # #undersampling
# # No_class = df[df["is_fraud"]==0]
# # yes_class = df[df["is_fraud"]==1]

# # No_class = resample(No_class, replace=False, n_samples=len(yes_class))
# # down_samples = pd.concat([yes_class, No_class], axis=0)

# # X = down_samples.drop('is_fraud', axis=1) 
# # y = down_samples['is_fraud'] 
# # test_size=0.25
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
# #X_test.to_csv('X_test.csv', index=False)
# #new=pd.Series(y_train).value_counts()
# df4=pd.read_csv('X_test.csv')
# df2=pd.read_csv('forestapp/fraudTest.csv')
# df3=pd.read_csv('combined_file.csv')

# columns_to_drop = ['first', 'unix_time', 'dob', 'cc_num', 'zip', 'city','street', 'state', 'trans_num', 'trans_date_trans_time']
# df=df2.drop(columns_to_drop,axis=1)
# df['merchant'] = df['merchant'].apply(lambda x : x.replace('fraud_',''))

# df['gender'] = df['gender'].map({'F': 0, 'M': 1})

# # applying WOE encoding
# for col in ['job','merchant', 'category', 'lat', 'last']:
#     df[col] = WOEEncoder().fit_transform(df[col],df['is_fraud'])

# df = df[df["is_fraud"]==1]
# # loaded_model = joblib.load('forestapp/new models/trainedmodel_labelled_undersampled_best_performing_20240310122606_.joblib')
# # X_test_point = df.iloc[2].values.reshape(1, -1)

# df['first']=df2['first']
# df['merchant_name']=df2['merchant']

# df.to_csv('frauddata.csv', mode='a', header=not os.path.isfile('frauddata.csv'), index=False)
# print(df.head())
# df=pd.read_csv('frauddata.csv')
# df=df.drop('is_fraud',axis=1)
# df.to_csv('frauddat.csv', mode='a', header=not os.path.isfile('frauddat.csv'), index=False)

# # X_test_point = X_test[2].reshape(1, -1)

# # # Make a prediction and get contributions
# prediction, bias, contributions = ti.predict(loaded_model, X_test_point)

# columns=df.columns
# feature_contributions = dict(zip(columns, contributions[0]))
# # contributions now contains the contribution of each feature to the prediction
# print("Prediction:", prediction)
# print("Bias (average prediction):", bias)
# print("Feature Contributions:", contributions)
# print(feature_contributions)
# prediction_class_0 = bias[0][0] + contributions[0][0].sum(axis=0)
# prediction_class_1 = bias[0][1] + contributions[0][1].sum(axis=0)
# print(prediction_class_0)
# print(prediction_class_1)
