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

### Problem Formulation

  * Input: diffferent features for residential house
  * Output: predicting sales price based on the different feature
  * Models used
    * Linear regression
    * Lasso
    * Ridge

### Training

  * sklearn learn, linear_model from sklearn
  * The run time was pertty quick, since the data set is not too large. 
  * Most of the difficulties come from the data cleaning and data preprocessing before fit to model. Model selection is also trick.
  
  
### Performance Comparison

* The metric is Root Mean Squared Error (RMSE).
![image](https://user-images.githubusercontent.com/89665013/207662557-01cb0b57-6774-48c4-ba21-63685aebf0b0.png)
 ![image](https://user-images.githubusercontent.com/89665013/207664625-27cdc49c-313c-419c-8973-3805d3acd081.png)
![image](https://user-images.githubusercontent.com/89665013/207664677-c9ee3b55-8c38-41e2-a0d0-b6080cc8f221.png)

### Conclusions

* The linear regression is the better model to using out of the 3, since the RMSE is smaller compare to the orders.

### Future Work

* I want to try using other models to see if that RMSE could be futher reduced.
* Use this method and apply this model to analysis and make prediction about house sale price locally.

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
* train.csv - the training set
* test.csv - the test set
* data_description.txt - full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here
* sample_submission.csv - a benchmark submission from a linear regression on year and month of sale, lot square footage, and number of bedrooms

### Software Setup
* ![image](https://user-images.githubusercontent.com/89665013/207668164-72b2955b-dafe-4d47-9c3f-80a7bc211b13.png)
* ![image](https://user-images.githubusercontent.com/89665013/207668256-03900f55-f810-4d8f-a9ea-2ce81f43af1e.png)
* ![image](https://user-images.githubusercontent.com/89665013/207668382-2bdbcf18-8fda-4fe1-b92a-8ff1a80ace90.png)

### Data

* Data could be downloaded in Kaggle: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data


## Citations

* https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview
* https://scikit-learn.org/stable/modules/linear_model.html
* https://www.kaggle.com/code/timolee/a-home-for-pandas-and-sklearn-beginner-how-tos
* https://www.youtube.com/watch?v=_-UCcuB8nbw
