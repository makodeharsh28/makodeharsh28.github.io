#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe
from flask import Flask
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")
#import matplotlib.pyplot as plt
from datetime import date
from pandas_datareader import data as pdr

df=pd.read_csv('datasets/Machine.csv')
df1=pd.read_csv('datasets/Crude_oil.csv')
df2=pd.read_csv('datasets/Textile.csv')


import pandas as pd

def select_dataset():
    print("Choose Industry Type:")
    print("1. Machinery and Equipment")
    print("2. Crude Oil and Petroleum")
    print("3. Textile")

    choice = input("Enter the number of the Industry you want to select: ")

    if choice == '1':
        return df
    elif choice == '2':
        return df1
    elif choice == '3':
        return df2
    else:
        print("Invalid choice. Please enter a number between 1 and 3.")
        return select_dataset()

# Example usage:
selected_dataset = select_dataset()


X=selected_dataset.drop("Category",axis='columns')
y=selected_dataset['Category']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
Risk_estimator=DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
Risk_estimator.fit(X_train, y_train)

y_predicted=Risk_estimator.predict(X_test)

df_ts=pd.read_csv('datasets\Machine_TS.csv', index_col='Year', parse_dates=True, squeeze=True, encoding='latin-1')
df_ts_1=pd.read_csv('datasets\Crude_TS.csv', index_col='Year', parse_dates=True, squeeze=True, encoding='latin-1')
df_ts_2=pd.read_csv('datasets\Textile_TS.csv', index_col='Year', parse_dates=True, squeeze=True, encoding='latin-1')


def select_dataset_ts():
    print("Choose Industry Type:")
    print("1. Machinery and Equipment")
    print("2. Crude Oil and Petroleum")
    print("3. Textile")

    choice = input("Enter the number of the Industry you want to select: ")

    if choice == '1':
        return df_ts
    elif choice == '2':
        return df_ts_1
    elif choice == '3':
        return df_ts_2
    else:
        print("Invalid choice. Please enter a number between 1 and 3.")
        return select_dataset_ts()

# Example usage:
selected__ts_dataset = select_dataset_ts()


original_dataset = pd.DataFrame(selected__ts_dataset)

# Slice the dataset into 7 different datasets (one for each column)
column_datasets = [original_dataset.iloc[:, i:i+1] for i in range(original_dataset.shape[1])]

# Slice the dataset into 7 different datasets (one for each column)
column_datasets = {f'dataset{i+1}': original_dataset.iloc[:, i:i+1] for i in range(original_dataset.shape[1])}



def user_input():
    n1 = int(input("Enter Raw material cost (300-1000 per unit): "))
    n2 = float(input("Enter Currency Exchange rate in dollar: "))
    n3 = float(input("Enter global demand of this year in percent: "))
    n4 = float(input("Enter Labor Cost (1000-2200 per unit): "))
    n5 = int(input("Enter Interest of current year(2-10): "))
    n6 = int(input("Enter consumer price Index(220-480): "))
    return np.array([n1, n2, n3, n4, n5, n6]).reshape(1, -1)

def main():
    user_data = user_input()
    output = Risk_estimator.predict(user_data)

    if output == 0:
        print("The borrower will not claim the Insurance")
    else:
        print("The borrower will claim the Insurance")


b=Risk_estimator.predict_proba(final)

pickle.dump(Risk_estimator,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


@app.route('/predict', methods=['POST','GET'])
def predict():


if __name__ == '__main__':
    app.run(debug=True)
