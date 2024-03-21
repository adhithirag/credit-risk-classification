# credit-risk-classification

## Overview of the Analysis

 - The purpose of this analysis is to train and evaluate a machine learning model to determine if this model can accurately predict the status of a loan based on loan risk. The dataset that was used in this analysis shows historical lending activity from a peer-to-peer lending services company. The basic information included in the dataset was the loan size, the interest rate, the income of the borrower, the debt to income ratio, total debt, and the loan status. 
 - For the purpose of developing our machine learning model, the column of interest in the dataset was the loan status. The values in this column were either an '0', which indicates that the loan is healthy, which means that the borrower is able to pay back the loan, or a '1', which means that the loan is at risk of defaulting. We want to predict the accuracy of the model predicting the '1's in the loan status column as this indicates how accurate the model can predict loan risk.
 - The first step of the machine learning process is to import the necessary dependencies to build our model 
    - import numpy as np
    - import pandas as pd
    - import matplotlib.pyplot as plt
    - from pathlib import Path
    - from sklearn.model_selection import train_test_split
    - from sklearn.metrics import confusion_matrix, classification_report
 - The second step is to load the lending_data.csv file and create a dataframe and then separate the data into our labels and features
    - the y-variable is our labels or what we are trying to predict using our machine learning model, which is the loan_status column
    - the x-variable are our features which are the rest of the columns in the dataframe, minus the loan_status column
 - The third step is to split the dataset into training and testing datasets. We will split the dataset on an 80:20 ration; this means that 80% of the dataset will be our training data and 20% of our dataset will be our testing data. To test the success of our machine learning model, our model needs to test data that it has not seen during training, which is why 20% of the dataset has been allocated for testing
    - The fourth step is to create a Logistic Regression model using the LogisticRegression module from the SKlearn library on our training data (X_train and y_train)
    - The fifth step is to make predictions using the testing data
    - The sixth step is to evaluate the model's performance by generating a confusion matrix that shows the true negative, false negative, false positive, and true positive values of our model
    - Lastly, we generated a classification report to get the accuracy score as well as the precision and recall scores. 

## Results

* Machine Learning Model 1: Logistic Regression
    * Description of Model 1 Accuracy, Precision, and Recall scores.

    - The accuracy of this model is 99%, which means that the ratio of correctly predicted observations divided by the total predicted positive observations is 0.99. This means that the model is able to accurately predict the number of "healthy loans".
    - The precision score is 1.00, which means that there is a low false positive rate as the model was able to predict correctly the number of healthy loans. In this example, false positive means that the model is predicting the risk of defaulting, or '1' but the actual loan status is that it is healthy. True positive means that the model is predicting that the loan is at risk, and the loan status is actually risk. We are getting an increased precision of the model predicting whether the loan status is actually at risk of defaulting.
    - The recall score is also 1.00 which means that the model gave us a low false negative rate. False negative in this example means that the model is predicting a healthy loan, but the loan is actually at risk of defaulting.

## Summary

* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

- The linear regression model seems to perform the best because we got a high accuracy score of 0.99, and we also got a high precision and high recall value. This means that based on the training data, the model can accurately predict that the loan is healthy and can also accurately predict whether the loan is at a high risk of defaulting. The performan is definitely dependent on the the problem that we are trying to solve. In this case, I think it is more important to predict the '1' 's because the goal is to determine to whom the lending company will lend money to and the company will want to loan money to those borrowers who will definitely be able to pay back the loan. Therefore, if the model accurately predicts which loans are high risk, then they won't loan money to those particular borrowers. The goal is to predict the creditworthiness of the borrowers by predicting the '1' 's. 