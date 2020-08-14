# Credit Card Default Binary Classification
This repo contains a process to predict whether a person will default on their credit card or not.  

## Repo Structure
**010.Process.ipynb**: a Jupyter notebook containing the entire process
**holdout_prediction.ipynb**: a Jupyter notebook containing codes for making a prediction using the final model (random forest)
**feature_transformation.py**: a code to transform feature data

## Data 
Default of credit card clients Data Set from UCI (https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). 

## Process
### Data Cleaning & EDA
#### Default (Target)
Our data shows that we have a class imbalance with only approximately 26% of the dataset being default class. We will test different oversampling and weighting options to control for this problem.

#### Credit Limit
As one might expect, our data is highly right skewed with a few extremely high credit limits.  
This skew is even more significant when it comes to default class. There are proportionately more people defaulting at a lower tier credit limit than higher credit limit. 

#### Sex
Our dataset contains more females than males. (13571 to 8928)  
The ratios of defaults per each sex don't differ too much.  

#### Education
There are some values that we are not able to derive the details of. Since our purpose is not necessarily to interprete the features but to predict, I will leave them as categorical values.

#### Marriage
There are also a value that we are not able to derive the details of. I'll leave them as is.

#### Age
Right skewed distribution with most people being in their 20s and 40s. There aren't too much difference between classes based on the distribution.

#### Payment History
There are more values than what codebook indicated, there's no way to assume the index system. I'll also treat them as categorical values. 

#### Billing Amount & Payment
There are total 6 billing amount and payment info. In order to figure out whether billing occurred first or payment occurred first in each month, I compared the number of times billing amount was equal to payment info for the same month and between previous month and current month. Based on that, I made an assumption that billing occurred after payment for each month. 

### Feature Engineering
Since most of the values were for each month, I figured they would be too highly correlated for default class but not for not-defaulted class. So instead of using the values as is, I engineered many new features. 

#### Payment History 
Instead of the payment history for each consecutive month, I wanted to see the tendency to be late or early. So I added up the number of times each categories of payment history occurred.

#### Average Balance
I calculated the balance for each month by subtracting the payment amount from the billing from previous month. Then I averaged the balance from the previous 6 months.

#### Cumulative change of balance
Then I also calculated how balance changed from month to month and added them up in order to find clients whose balance continuously increased or decreased. 

#### Final balance and payment
I kept the last bill amount and payment amount as the final balance and payment. 

#### Number of Underpayment
I calculated how many times each client paid less than minimum payment. I used $35 as a minimum payment amount.

#### Over Credit Limit
I wanted to see how often clients tend to spend over their credit limit. So I calculated the number of times their bill amount was higher than their credit limit.

#### Percent Usages
I also calculated on average what percentage of their credit limit each client tend to spend. 

#### Average percent payment
Additionally, I added a feature that shows on average what percentage of billed amount they tend to pay. 

#### Payment Pattern Changes
IThen I used a difference between their first and last percent payment to denote the change in payment pattern.

#### Max Bill Amount
I also included the maximum bill amount each client had. (But this highly correlated with other measures, so got removed later on)

#### Existing payments
I kept the last 2 payment history and whether they paid less than minimum or not in the last 2 months.

### Data Standardization & Nomalization
I log-transformed the credit limit so it approximates normality.   
Also I treated sex, education, marraige, and payment history (whether they paid ontime or delayed by n number of months) of last two months as categorical and turned them into dummies. 
All values were standardized using StandardScaler fro Sci-kit Learn.

### Interactions
I iterated through all pairs of standardized dataset to find the top 10 interactions that contribute to the logistic regression using KFold and cross_val_score. 

A number of some types of payment history tend to interact with payment history of the last two months. Since I don't know exactly what the payment history rankings mean, I cannot really interpret these accurately.

## Model Evaluation & Selection
First, I split the final dataset into train and test sets. (75/25)  

### Class Imbalance Problem
I tested a balance weight method, SMOTE, and Tomek Links and their performance on logistic regression. Tomek Links led to highly overfit model. SMOTE and the balance weight method both performed similarly well. I chose the balance weight method. 

### Logistic Regression
For logistic regression, I needed to take care of multicollinearity issues. For this, I tried manually controlling for the highly correlated measures, using Variance Inflation Factor score, and Lasso regularization for the feature selection. Lambda value for Lasso was chosen using GridSearchCV.

Manually controlling yielded the best performance (Test F1 = 0.516) , followed by Lasso (C = 0.5, F1 = 0.445), then VIF (F1 = 0.437). 

### KNN
For K-nearest neighbors, I ran GridSearchCV for different k-values. At K = 21, the model performed the best, but not as well as manually controlled logistic regression or Lasso regression. (F1 = 0.439)

### A simple decision tree
Using SMOTE resampled dataset, I tested the decision tree. Hyperparameters were chosen using GridSearchCV for max_depth, min_samples_split, min_samples_leaf, and criterion. The model using gini index at max_depth 7, min_samples_leaf 4, min_samples_split 9 yielded the best model with F1 = 0.508. This model showed a trend of overfitting. (Training F1 > 0.76)

### Random Forest
Then I ran GridSearchCV to fit the random forest on the best hyperparameters. A model with balanced class-weight and entropy criterion, max_depth of 7, 30% max feature and 3 minimum samples to split yielded the best model with F1 = 0.533. This was chosen as the final model.

### XGBoost
XG boost was also tested. Even though XGboost result yieded very high training accuracy, it's generalizability was rather low. Reducing the max_depth from the best hyperparameters eased this problem but still was not as good as random forest. 





