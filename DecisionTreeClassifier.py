import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.preprocessing import LabelEncoder

# Function importing Dataset 
def importdata(): 
    balance_data = pd.read_csv('dermatology.csv', sep=',', header=None) 
      
    # Encoding categorical data into numeric 
    labelencoder = LabelEncoder()
    for column in balance_data.columns:
        balance_data[column] = labelencoder.fit_transform(balance_data[column])
      
    print("Dataset Length: ", len(balance_data)) 
    print("Dataset Shape: ", balance_data.shape) 
    print("Dataset: ", balance_data.head()) 
    return balance_data 
  
# Function to split the dataset 
def splitdataset(balance_data): 
    # Separating the target variable 
    X = balance_data.values[:, :-1] 
    Y = balance_data.values[:, -1] 
  
    # Splitting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=140) #0.1,140
    return X, Y, X_train, X_test, y_train, y_test 
      
# Function to perform training with giniIndex 
def train_using_gini(X_train, y_train): 
    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5) 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
      
# Function to perform training with entropy 
def train_using_entropy(X_train, y_train): 
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5) 
    clf_entropy.fit(X_train, y_train) 
    return clf_entropy 
  
# Function to make predictions 
def prediction(X_test, clf_object): 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
      
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred)) 
    print("Accuracy: ", accuracy_score(y_test, y_pred) * 100) 
    print("Report:\n", classification_report(y_test, y_pred)) 
  
# Driver code 
def main(): 
    # Building Phase 
    data = importdata() 
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data) 
    clf_gini = train_using_gini(X_train, y_train) 
    clf_entropy = train_using_entropy(X_train, y_train) 
      
    # Operational Phase 
    print("Results Using Gini Index:") 
    y_pred_gini = prediction(X_test, clf_gini) 
    cal_accuracy(y_test, y_pred_gini) 
      
    print("Results Using Entropy:") 
    y_pred_entropy = prediction(X_test, clf_entropy) 
    cal_accuracy(y_test, y_pred_entropy) 

# Calling main function 
if __name__ == "__main__": 
    main()
