import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("loan_prediction.csv")

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


final_data.drop(cat_cols, axis="columns", inplace=True)


features = final_data.drop(["Loan_ID", "Loan_Status"], axis="columns")
target = final_data["Loan_Status"]


x_train, x_test, y_train, y_test = train_test_split(features, target)

model = RandomForestClassifier()

model.fit(x_train, y_train)
print("Accuracy is", model.score(x_test, y_test)*100)

import pickle
file=open("model.pkl", 'wb')
pickle.dump(model, file)



