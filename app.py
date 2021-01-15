from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import pickle

from os import path as os_path
path = os_path.abspath(os_path.split(__file__)[0])
app = Flask(__name__,template_folder='template')

@app.route('/')
def index():
  return render_template('index.html')
@app.route('/test')
def test():
  data = pd.read_excel(path+"\\Telco_customer_churn.csv.xlsx")
  rows_to_delete = [i for i in data.index if data['Total Charges'][i]==" "]
  data.drop(rows_to_delete, inplace=True)
  data['Total Charges'] = pd.to_numeric(data['Total Charges'], downcast="float")
  oh_encoded_cols = ['Phone Service', 'Multiple Lines','Payment Method', 
                      'Internet Service', 'Online Security', 
                      'Online Backup', 'Device Protection', 
                      'Tech Support','Contract','Paperless Billing']
  label_encoded_cols = ['Senior Citizen', 'Partner', 'Dependents']
  n_X_train = data[['Phone Service', 'Multiple Lines','Payment Method', 
                      'Internet Service', 'Online Security', 
                      'Online Backup', 'Device Protection', 
                      'Tech Support','Contract','Paperless Billing','Senior Citizen', 'Partner', 'Dependents', 'Total Charges', 'Tenure Months', 'Monthly Charges' ]]
  X_train = data[['Phone Service', 'Multiple Lines','Payment Method', 
                      'Internet Service', 'Online Security', 
                      'Online Backup', 'Device Protection', 
                      'Tech Support','Contract','Paperless Billing','Senior Citizen', 'Partner', 'Dependents', 'Total Charges', 'Tenure Months', 'Monthly Charges' ]]  
                                      
  ohe= pickle.load(open(path+'\\oneHotEncoder.pickle','rb'))
  le =LabelEncoder()
  le.fit(['No','Yes'])
  X_train['Senior Citizen']=le.transform(X_train['Senior Citizen'])
  X_train[['Partner']]=le.transform(X_train[['Partner']])
  X_train[['Dependents']]=le.transform(X_train[['Dependents']])
  
  X_train_new = ohe.transform(X_train[oh_encoded_cols])

  col_names = ohe.get_feature_names(oh_encoded_cols)

  new_train_cols = pd.DataFrame(X_train_new, columns=col_names)

  new_train_cols.index = X_train.index

  train_num_cols = X_train.drop(oh_encoded_cols, axis = 1)

  X_train = pd.concat([train_num_cols,new_train_cols], axis=1)

  X_train.index = np.arange(X_train.shape[0])
 
  ad = pickle.load(open(path+'\\adModel.pickle','rb'))
  
  y_pred = ad.predict(X_train[['Senior Citizen', 'Partner', 'Dependents', 'Tenure Months',
       'Monthly Charges', 'Total Charges', 'Multiple Lines_No',
       'Multiple Lines_Yes', 'Payment Method_Bank transfer (automatic)',
       'Payment Method_Credit card (automatic)',
       'Payment Method_Electronic check', 'Internet Service_DSL',
       'Internet Service_Fiber optic', 'Internet Service_No',
       'Online Security_No', 'Online Security_Yes', 'Online Backup_No',
       'Online Backup_Yes', 'Device Protection_No', 'Device Protection_Yes',
       'Tech Support_No', 'Tech Support_Yes', 'Contract_Month-to-month',
       'Contract_One year', 'Paperless Billing_Yes']])
  print(y_pred)
  return str(np.count_nonzero(y_pred == 1))

@app.route('/formulaire',methods = ['POST'])
def resultat():
  result = request.form
  Total_Charges = float(result['Total_Charges'])
  Tenure_Months = int(result['Tenure_Months'])
  Monthly_Charges =  float(result['Monthly_Charges'])

  Payment_Method = result['Payment_Method']

  Internet_Service = result['Internet_Service']
  Online_Security = result['Online_Security']
  Online_Backup = result['Online_Backup']
  Device_Protection = result['Device_Protection']
  Tech_Support = result['Tech_Support']
  Multiple_Lines = result['Multiple_Lines']
  
  Contract = result['Contract']
  
  Paperless_Billing = result['Paperless_Billing']

  Phone_Service = result['Phone_Service']

  Senior_Citizen = result['Senior_Citizen']
  Partner = result['Partner']
  Dependents = result['Dependents']

  oh_encoded_cols = ['Phone Service', 'Multiple Lines','Payment Method', 
                      'Internet Service', 'Online Security', 
                      'Online Backup', 'Device Protection', 
                      'Tech Support','Contract','Paperless Billing']
  label_encoded_cols = ['Senior Citizen', 'Partner', 'Dependents']

  num_col=pd.DataFrame([[Total_Charges,Tenure_Months,Monthly_Charges]],columns=['Tenure Months','Monthly Charges','Total Charges'])
  cat_col_ord=pd.DataFrame([[Phone_Service,Senior_Citizen,Partner,Dependents]],columns=["Phone Service",  "Senior Citizen", "Partner", "Dependents"])
  cat_col_one=pd.DataFrame([[Paperless_Billing,Payment_Method,Multiple_Lines,Internet_Service,Online_Security,Online_Backup,Device_Protection,Tech_Support,Contract ]],columns=['Paperless Billing','Payment Method','Multiple Lines','Internet Service','Online Security','Online Backup','Device Protection','Tech Support','Contract'])
  X_train=pd.concat([num_col,cat_col_ord,cat_col_one], axis="columns")

  oh_encoded_cols = ['Phone Service', 'Multiple Lines','Payment Method', 
                      'Internet Service', 'Online Security', 
                      'Online Backup', 'Device Protection', 
                      'Tech Support','Contract','Paperless Billing']
  label_encoded_cols = ['Senior Citizen', 'Partner', 'Dependents']
  n_X_train = X_train.copy()
  ohe= pickle.load(open(path+'\\oneHotEncoder.pickle','rb'))
  le =LabelEncoder()
  le.fit(['No','Yes'])
  print(cat_col_ord[['Senior Citizen']])
  X_train[['Senior Citizen']]=le.transform(X_train[['Senior Citizen']])
  X_train[['Partner']]=le.transform(X_train[['Partner']])
  X_train[['Dependents']]=le.transform(X_train[['Dependents']])
  
  X_train_new = ohe.transform(X_train[oh_encoded_cols])

  col_names = ohe.get_feature_names(oh_encoded_cols)

  new_train_cols = pd.DataFrame(X_train_new, columns=col_names)

  new_train_cols.index = X_train.index

  train_num_cols = X_train.drop(oh_encoded_cols, axis = 1)

  X_train = pd.concat([train_num_cols,new_train_cols], axis=1)

  X_train.index = np.arange(X_train.shape[0])
 
  ad = pickle.load(open(path+'\\adModel.pickle','rb'))
  
  y_pred = ad.predict(X_train[['Senior Citizen', 'Partner', 'Dependents', 'Tenure Months',
       'Monthly Charges', 'Total Charges', 'Multiple Lines_No',
       'Multiple Lines_Yes', 'Payment Method_Bank transfer (automatic)',
       'Payment Method_Credit card (automatic)',
       'Payment Method_Electronic check', 'Internet Service_DSL',
       'Internet Service_Fiber optic', 'Internet Service_No',
       'Online Security_No', 'Online Security_Yes', 'Online Backup_No',
       'Online Backup_Yes', 'Device Protection_No', 'Device Protection_Yes',
       'Tech Support_No', 'Tech Support_Yes', 'Contract_Month-to-month',
       'Contract_One year', 'Paperless Billing_Yes']])
  if y_pred==0:
      return render_template('still.html')
  elif y_pred==1:
      return render_template('leave.html')

         
app.run(debug=True)