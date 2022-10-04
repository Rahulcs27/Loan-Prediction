import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = pd.read_csv("loan_prediction.csv")
#print(data.head())
#print(data.tail())
#print(data.shape)

print(data.isnull().sum())

new_data = data.fillna({
	"Gender" : "Male",
	"Married": "Yes",
	"Dependents": 0,
	"Self_Employed": "No",
	"LoanAmount" : data["LoanAmount"].mean(),
	"Loan_Amount_Term": data["Loan_Amount_Term"].mean(),
	"Credit_History": 1
})
print(new_data.isnull().sum())

cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
dummies = pd.get_dummies(data[cat_cols], drop_first=True)

final_data = pd.concat([new_data, dummies], axis="columns")
#print(final_data.head())

final_data.drop(cat_cols, axis="columns", inplace=True)
#print(final_data.head())
#final_data.to_csv("f1.csv")

features = final_data.drop(["Loan_ID", "Loan_Status"], axis="columns")
target = final_data["Loan_Status"]

#print(features.head())
#print(target.head())

x_train, x_test, y_train, y_test = train_test_split(features, target)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
cr = classification_report(y_test, y_pred)
#print(cr)


#data = [[4583,1508,128,360,1,1,1,1,0,0,0,0,0,0]]
#data = [[1853,2840,1140000,360,1,1,0,0,0,0,0,0,0,0]]
#res = model.predict(data)
#print(res)

score = model.score(x_test, y_test)
print(score)

from sklearn.tree import DecisionTreeClassifier
model2 = DecisionTreeClassifier()
model2.fit(x_train, y_train)
print("Accuracy is", model2.score(x_test, y_test)*100)

from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier()

model3.fit(x_train, y_train)
print("Accuracy is", model3.score(x_test, y_test)*100)

import pickle
file=open("model.pkl", 'wb')
pickle.dump(model3, file)



