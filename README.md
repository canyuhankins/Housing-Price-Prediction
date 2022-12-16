# Housing Price Prediction - Kaggle Challenge

This repository holds an attempt to estimate the price of houses based on the given attributes that describes the aspect of residential homes in Ames, Iowa. https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

## Overview
The task is to estimate the price of houses based on the given 79 attributes that describes the aspect of residential homes in a area. The goal is to predict the sales price of each house and submission are based on Root Mean Squared Error (RMSE) between the logarithm of the prediction and the logarithm of the real sales price. I use the linear regression, ridge and lasso method and turns out the linear regression is my best model, with a score of 0.73686, and the closer to 0 and better fit of the model.

## Data
* Data:
  * Type: 
    * Input: CSV file: description of a residential house; Training and test set
  * Size: 957.39 kB
  * The instances include 1460 points given for training dataset and 1459 given for testing dataset.
  * ![image](https://user-images.githubusercontent.com/89665013/208147103-35dfa13f-68ca-40a8-bbe7-009c7493c86a.png)
  * Historgram represents the sales price of the train dataset, it's normally distributed with some outlier on the right, also it's skewed to right due to the outlier.

#### Preprocessing / Clean up

* Dealing with not a number (NaN)
  * Drop the columns when NaN values are greater than 80% of the entire dataset.
  * With numerical variables, replace with mean
  * With categorical variables, replace with mode
* Not every NaN is a missing value! 
  * In can mean as an valid input as well
  * For example: Alley column have inputs, Grvl: Gravel, Pave: Paved, and NA: No alley access.
  * Need to go through and check them 
* One-hot encoding for categorical variables

### Problem Formulation

  * Input: diffferent features for residential house
  * Output: predicting sales price based on the input
  * Models used: 
    * Linear regression
    * Lasso
    * Ridge

### Training

  * sklearn learn, linear_model from sklearn
  * The run time was pertty quick, since the data set is not too large. 
  * Most of the difficulties come from the data cleaning and data preprocessing before fit to model. Model selection is also trick.
  
  
### Performance Comparison

* The metric is Root Mean Squared Error (RMSE):

![image](https://user-images.githubusercontent.com/89665013/207662557-01cb0b57-6774-48c4-ba21-63685aebf0b0.png)
![image](https://user-images.githubusercontent.com/89665013/207692121-30d2a03c-bdf1-4826-8c25-8107686eaf7d.png)

### Conclusions

* The linear regression is the better model to using out of the 3, since the RMSE is smaller compare to the orders. It was a lot of fun doing this challenge, I had a hard time setting up the data and also cleaning the data, I think that part take most of my time, because I want to avoid as much of error as possible, so that requires some thinking and trying out different things to see which method/way is best to do.

### Future Work

* I would like to see what will be different when I clean up the data in some other way, for example, dropping the rows with 80% missing value, or replace them with 0, or dropping everything that have a missing value. Compare to what I have done: dropping the columns with more than 80% of the missing values instead of the rows.
* I want to try using other models to see if that RMSE could be futher reduced, like random forest, or even other machine learning methods like kera.
* I can further use this model and apply them to analyze and make prediction about the local sale price for residential houses.

## How to reproduce results

* Set up the enviroment and install nesscessary packages: numpy, pandas, matplotlib, sklearn)
* Data cleaning:
 * Combine test and train data set, for better average per column etc.
 * Drop a column if missing value is greater than 80% of the dataset.
 * Go through each variable carefully and clearly define the input within each cell.
  * With numerical variables, replace with mean
  * With categorical variables, replace with mode
 *Not every NaN is a missing value! 
  *In can mean as an valid input as well. For example: Alley column have inputs, Grvl: Gravel, Pave: Paved, and NA: No alley access.
* One-hot encoding for categorical variables
* Split the combine dataset to train and sets
* Fit into linear model
* Calculate the metrics, and scores.

### Overview of files in repository

* Project.ipynb - the code to process the data and make analysis on the given dataset.
* house-prices-advanced-regression-techniques (from kaggle)
 * train.csv - the training set
 * test.csv - the test set
 * data_description.txt - full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here
 * sample_submission.csv - a benchmark submission from a linear regression on year and month of sale, lot square footage, and number of bedrooms


### Software Setup
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error 
from IPython.display import HTML, display
import tabulate

### Data

* Data could be downloaded in Kaggle: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data


## Citations

* https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview
* https://scikit-learn.org/stable/modules/linear_model.html
* https://www.kaggle.com/code/timolee/a-home-for-pandas-and-sklearn-beginner-how-tos
* https://www.kaggle.com/code/thoolihan/housing-price-prediction-w-sklearn
* https://www.youtube.com/watch?v=_-UCcuB8nbw
