# Logistic Regression

Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model. It is particularly useful for situations where the outcome is binary.

## Fundamental Concepts and Formulas

Logistic regression predicts the probability that a given input belongs to the class labeled "1". The probabilities predicting class "1" are computed with the logistic function, which is defined as:

P(y = 1 | x) = 1 / (1 + exp(-z))

where z is the linear combination of the input features x weighted by the coefficients β derived from the training data:

z = β0 + β1x1 + β2x2 + ... + βnxn

The coefficients are estimated using maximum likelihood estimation (MLE), which aims to find the parameter values that maximize the likelihood of the observed sample.

## Implementation 

#### Libraries
```python
# Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from utils import *
```

#### Prepare the Dataset

Read data

```python
df = pd.read_csv('data.csv', sep = "\t")
df.head()
```

Convert target variable as categorical

```python
df.Y = df.Y.astype('category')
df.info()
```

Define input and output matrices

```python
INPUTS = ['X1','X2']
OUTPUT = 'Y'
X = df[INPUTS]
y = df[OUTPUT]
```

Train Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,  #percentage of test data
                                                    random_state=0, #seed for replication
                                                    stratify = y)   #Preserves distribution of y
```

Create dataset to store model predictions

```python
dfTR_eval = X_train.copy()
dfTR_eval['Y'] = y_train
dfTS_eval = X_test.copy()
dfTS_eval['Y'] = y_test
```

## **Logistic Regression with linear variables**

```python
INPUTS_LR = INPUTS

LogReg_fit = Pipeline(steps=[('scaler',StandardScaler()), # Preprocess the variables when training the model 
                        ('LogReg',LogisticRegression(penalty=None))]) # Model to use in the pipeline

LogReg_fit.fit(X_train[INPUTS_LR], y_train)

print("Logistic Regression Trained")
```

Cross Validation Score
```python
cross_validation_score = cross_val_score(LogReg_fit, X_train[INPUTS_LR], y_train, cv=10, scoring='accuracy').mean()
print(f"CV accuracy is {cross_validation_score}")
```

Summary Logistic Regression

```python
summaryLogReg(LogReg_fit, X_train[INPUTS_LR], y_train)
```

Model report on predictions
```python
## Obtain a report of the model based on predictions
dfTR_eval['Y_LR_pred'] = LogReg_fit.predict(X_train[INPUTS_LR])
dfTR_eval['Y_LR_prob_neg'] = LogReg_fit.predict_proba(X_train[INPUTS_LR])[:,0]
dfTR_eval['Y_LR_prob_pos'] = LogReg_fit.predict_proba(X_train[INPUTS_LR])[:,1]

### Scale test using preprocess in training
dfTS_eval['Y_LR_pred'] = LogReg_fit.predict(X_test)
dfTS_eval['Y_LR_prob_neg'] = LogReg_fit.predict_proba(X_test[INPUTS_LR])[:,0]
dfTS_eval['Y_LR_prob_pos'] = LogReg_fit.predict_proba(X_test[INPUTS_LR])[:,1]

#visualize evaluated data
dfTR_eval.head()
```

Plot Classification in a 2 dimensional space
```python
plot2DClass(X_train[INPUTS_LR], dfTR_eval['Y'], LogReg_fit, 'X1', 'X2', 'YES', 50) 
```

**Train & Test Confusion Matrices**

```python
print("----- TRAINING CONFUSION MATRIX -----")
confusion_matrix(dfTR_eval['Y'], dfTR_eval['Y_LR_pred'],labels=['NO','YES'])
```

```python
print("----- TEST CONFUSION MATRIX-----")
confusion_matrix(dfTS_eval['Y'], dfTS_eval['Y_LR_pred'],labels=['NO','YES'])
```

**Plot Class Performance**
- Calibration plot
- Probability of class
- ROC, Area under de ROC curve
- Accuracy across possible cutoffs

```python
plotClassPerformance(dfTR_eval['Y'], LogReg_fit.predict_proba(X_train[INPUTS_LR]), selClass='YES')
```

## **Logistic Regression with linear variables**

```python
INPUTS_LR2_NUM = ["X2"]
INPUTS_LR2_SQ = ["X1"]
INPUTS_LR2_CAT = []
INPUTS_LR2 = INPUTS_LR2_NUM + INPUTS_LR2_SQ + INPUTS_LR2_CAT

# Prepare the numeric variables by scaling
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# Prepare the numeric variables by scaling
square_transformer = Pipeline(steps=[('scaler', StandardScaler()),
                                      ('Poly',PolynomialFeatures())])

# Prepare the categorical variables by encoding the categories
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Create a preprocessor to perform the steps defined above
preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, INPUTS_LR2_NUM),
        ('sq', square_transformer, INPUTS_LR2_SQ),
        ('cat', categorical_transformer, INPUTS_LR2_CAT)
        ])

pipe = Pipeline(steps=[('Prep',preprocessor), 
                       ('LogReg',LogisticRegression(fit_intercept=False, penalty=None))])

nFolds = 10
param = {}
LogReg_fit_sq = GridSearchCV(estimator=pipe,
                            param_grid=param,
                            n_jobs=-1,
                            scoring='accuracy',
                            cv=nFolds)

LogReg_fit_sq.fit(X_train[INPUTS_LR2], y_train)

print("Logistic Regression Trained")
```

Cross Validation Score
```python
cross_validation_score = cross_val_score(LogReg_fit_sq, X_train[INPUTS_LR2], y_train, cv=10, scoring='accuracy').mean()
print(f"CV accuracy is {cross_validation_score}")
```

Summary Logistic Regression

```python
summaryLogReg(LogReg_fit_sq.best_estimator_, X_train[INPUTS_LR2], y_train)
```

Model report on predictions
```python
## Obtain a report of the model based on predictions ---------------------------------------------
dfTR_eval['Y_LR_sq_pred'] = LogReg_fit_sq.predict(X_train[INPUTS_LR2])
dfTR_eval['Y_LR_sq_prob_neg'] = LogReg_fit_sq.predict_proba(X_train[INPUTS_LR2])[:,0]
dfTR_eval['Y_LR_sq_prob_pos'] = LogReg_fit_sq.predict_proba(X_train[INPUTS_LR2])[:,1]
### Scale test using preprocess in training
dfTS_eval['Y_LR_sq_pred'] = LogReg_fit_sq.predict(X_test[INPUTS_LR2])
dfTS_eval['Y_LR_sq_prob_neg'] = LogReg_fit_sq.predict_proba(X_test[INPUTS_LR2])[:,0]
dfTS_eval['Y_LR_sq_prob_pos'] = LogReg_fit_sq.predict_proba(X_test[INPUTS_LR2])[:,1]

#visualize evaluated data
dfTR_eval.head()
```

Plot Classification in a 2 dimensional space
```python
plot2DClass(X_train[INPUTS_LR], dfTR_eval['Y'], LogReg_fit, 'X1', 'X2', 'YES', 50) 
```

**Train & Test Confusion Matrices**

```python
print("----- TRAINING CONFUSION MATRIX -----")
confusion_matrix(dfTR_eval['Y'], dfTR_eval['Y_LR_sq_pred'], labels=['NO','YES'])
```

```python
print("----- TEST CONFUSION MATRIX-----")
confusion_matrix(dfTS_eval['Y'], dfTS_eval['Y_LR_sq_pred'], labels=['NO','YES'])
```

**Plot Class Performance**
- Calibration plot
- Probability of class
- ROC, Area under de ROC curve
- Accuracy across possible cutoffs

```python
plotClassPerformance(dfTR_eval['Y'], LogReg_fit_sq.predict_proba(X_train[INPUTS_LR2]), selClass='YES')
```




