# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries.

2.Create a sample dataset with student features and placement status.

3.Split the dataset into training and testing sets.

4.Standardize the feature values.

5.Train the logistic regression model.

6.Predict placement status on test data.

7.Evaluate the model using accuracy score, confusion matrix, and classification report.

8.Predict placement status for a new student input.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Mahalakshmi S
RegisterNumber:  25018377
# Logistic Regression for Student Placement Prediction

# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
data = pd.read_csv("C:/Users/acer/Downloads/Placement_Data.csv")   

print("Dataset Preview:")
print(data.head())

#Drop Unnecessary Columns
data = data.drop(["sl_no", "salary"], axis=1)

#Convert Target Variable (status) to Binary
# Placed = 1, Not Placed = 0
data["status"] = data["status"].map({"Placed": 1, "Not Placed": 0})

#Separate Features and Target
X = data.drop("status", axis=1)
y = data["status"]

#One-Hot Encode Categorical Variables
X = pd.get_dummies(X, drop_first=True)

print("\nAfter Encoding:")
print(X.head())

#Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

#Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Make Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

#Model Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Placement Prediction")
plt.show()
*/
```

## Output:
<img width="824" height="824" alt="Screenshot 2026-02-04 094648" src="https://github.com/user-attachments/assets/6e64a44d-821b-4940-a083-78aa06809c1a" />
<img width="749" height="818" alt="Screenshot 2026-02-04 094709" src="https://github.com/user-attachments/assets/2ce5ad48-c80b-4a88-91f6-de5921fd0ec9" />




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
