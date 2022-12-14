# Housing Price Prediction - Kaggle Challenge

This repository holds an attempt to estimate the price of houses based on the given attributes that describes the aspect of residential homes in Ames, Iowa. https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

## Overview
The task is to estimate the price of houses based on the given 79 attributes that describes the aspect of residential homes in a area. The goal is to predict the sales price of each house and submission are based on Root Mean Squared Error (RMSE) between the logarithm of the prediction and the logarithm of the real sales price. I use the linear regression, ridge and lasso method and turns out the linear regression is my best model, with a score of 0.73686, and the closer to 0 and better fit of the model.

## Data
* Data:
  * Type: 
    * Input: CSV file: description of a residential house; Training and test set
  * Size: 957.39 kB
  * Instances 1460 points given for training and 1459 given for testing.

#### Preprocessing / Clean up

* Dealing with not a number (NaN)
  * Drop the columns when NaN values are greater than 80%
  * With numerical variables, replace with mean
  * With categorical variables, replace with mode
* Not every NaN is a missing value! 
  * In can mean as an valid input as well
  * For example: Alley column have inputs, Grvl: Gravel, Pave: Paved, and NA: No alley access.
  * Need to go through and check them 
* One-hot encoding for categorical variables

#### Data Visualization
The PCA (Principal Component Analysis) is shown: 
![image](https://user-images.githubusercontent.com/89665013/207643348-0a6afc21-dcb3-4963-8d84-c2fc35b42cac.png)
* Can use 100-150 features and variance is still significant

### Problem Formulation

* Define:
  * Input / Output
  * Models
    * Describe the different models you tried and why.
  * Loss, Optimizer, other Hyperparameters.
