#lets create the best performing model

#import basic libraries
import pandas as pd
import numpy as np

#read the file
data = pd.read_csv("loan_prediction.csv")

#data cleaning
data[data.select_dtypes(['object']).columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
data = data.drop(['Loan_ID'], axis = 1)
data['Credit_History'] = data['Credit_History'].astype('category')
data['Gender'] = data['Gender'].map({'Male':'0', 'Female':'1'})
data['Married'] = data['Married'].map({'No':'0', 'Yes':'1'})
data['Education'] = data['Education'].map({'Not Graduate':'0', 'Graduate':'1'})
data['Self_Employed'] = data['Self_Employed'].map({'No':'0', 'Yes':'1'})
data['Property_Area'] = data['Property_Area'].cat.codes
data['Dependents'] = data['Dependents'].map({'3+':'3','0':'0','1':'1','2':'2'})
data['Loan_Status'] = data['Loan_Status'].cat.codes

#since we have have outliers
#lets keep a maximum limit to quantitative variables
data['ApplicantIncome'].loc[(data['ApplicantIncome']> 30000)] = 30000
data['CoapplicantIncome'].loc[(data['CoapplicantIncome']> 20000)] = 20000
data['LoanAmount'].loc[(data['LoanAmount']> 600)] = 600

#missing value imputation
# categorical columns
data.fillna(data.select_dtypes(include='category').mode().iloc[0], inplace=True)
mean = round(data['LoanAmount'].mean())
data['LoanAmount'] = data['LoanAmount'].fillna(mean)
mean1 = round(data['Loan_Amount_Term'].mean())
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(mean1)
#standardization
#from sklearn.preprocessing import MinMaxScaler
#mms = MinMaxScaler()
#data['ApplicantIncome'] = mms.fit_transform(np.array(data['ApplicantIncome']).reshape(-1,1))
#data['CoapplicantIncome'] = mms.fit_transform(np.array(data['CoapplicantIncome']).reshape(-1,1))
#data['LoanAmount'] = mms.fit_transform(np.array(data['LoanAmount']).reshape(-1,1))
#data['Loan_Amount_Term'] = mms.fit_transform(np.array(data['Loan_Amount_Term']).reshape(-1,1))

#x-y
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
print(x.info())

#train-validation split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

models = {DecisionTreeClassifier,
          LogisticRegression, RandomForestClassifier,
          GaussianNB, KNeighborsClassifier}
for model in models:
    m = model().fit(X_train,y_train)
    predict = m.predict(X_test)
    c_r = classification_report(y_test,predict)
    print(model, "\n" ,c_r)
#Logistic regression is best algoritjmns in terms of accuracy

model_= LogisticRegression().fit(X_train, y_train)

#import pickle
#Saving model to disk
#pickle.dump(model_, open('model.pkl','wb'))

result = model_.predict([[0,1,1,1,1,30000,20000,500,240,0,2]])
import pickle
model = pickle.load(open('model.pkl', 'rb'))
results = model.predict([[0,1,1,1,1,30000,20000,500,240,0,2]])
print("____final_result", results)