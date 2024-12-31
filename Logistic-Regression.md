# Logistic Regression

Logistic regression machine learning is a key classification technique. It can work with both numerical and categorical data, making it versatile for various applications. For example, it’s commonly used to predict whether a customer will leave a service (churn), identify fraudulent transactions, or determine if a patient has a specific condition.

One of the main advantages of logistic regression is its simplicity. Logistic regression machine learning not only predicts outcomes but also helps understand which factors are most important for these predictions. This makes logistic regression a practical tool for solving classification problems while providing clear insights into the data

## Fundamental Concepts and Formulas

#### **Sigmoid Function**

The sigmoid or logistic function is essential for converting predicted values into probabilities in logistic regression. This function maps any real number to a value between 0 and 1, ensuring that predictions remain within this probability range. Its "S" shaped curve helps translate raw scores into a more interpretable format.

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

A threshold value is used in logistic regression to make decisions based on these probabilities. For instance, if the predicted probability is above a certain threshold, such as 0.5, the result is 1

#### **Types of Logistic Regression**

The main types of logistic regression include Binary Logistic Regression, Multinomial Logistic Regression, and Ordinal Logistic Regression.

- **Binary logistic regression** is the most common type of logistic regression, where the dependent variable has only two possible outcomes or classes, typically represented as 0 and 1
- **Multinomial Logistic Regression** also known as softmax regression, is used when the dependent variable has more than two unordered categories. It models the probability of an observation belonging to each class using the softmax function, which ensures that the predicted probabilities sum up to one across all classes.

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}$$
- **Ordinal Logistic Regression** is employed when the dependent variable has more than two ordered categories. In other words, the outcome variable has a natural ordering or hierarchy among its categories

#### Logistic Model Formula

$$p(x) = \frac{e^{w_0 + w_1 x}}{1 + e^{w_0 + w_1 x}}$$

- $$p(x)$$ represents the probability that Y = 1 given x
- $$w_0$$ and $$w_1$$ are model coefficients. The $$w_0$$ is the bias term, is a constant that provides an additional adjustment to the decision boundary of the model, it shifts the entire logistic function left or right. The $$w_1$$ are associated with individual features $$(x_1, x_2...)$$ they describe the impact of each feature on the predcition. A positive coefficient increases the probability that the outcome variable Y equals 1 as the feature increases, while a negative does the opposite. The significance of the of these coefficients are determined by the p-value in hypothesis testing. The regression coefficients are estimated by maximizing the likelihood function. 
- $$x$$ is the feature value

The odds formula, particularly in the context of logistic regression, is an expression that describes the ratio of the probability of an event occurring to the probability of it not occurring. In logistic regression, if we denote $$p(x)$$ as the probability that the outcome variable $$Y$$ is 1 given predictors x, then the odds of $$Y=1$$ are given by: 

$$\frac{p(x)}{1 - p(x)} = e^{w_0 + w_1x}$$

The quantity of $$\frac{p(x)}{1 - p(x)}$$ is called the odds.

The logistic regression model uses this odds concept to link the predictors with the outcome. Specifically, the model expresses the natural logarithm of the odds (log-odds) as a linear combination of the predictors

**Derivation of Logistic Regression Odds Formula**

1. **Start with the Logistic Regression Formula:**
   We begin with the logistic regression function for probability $$p(x)$$:
   $$p(x) = \frac{e^{w_0 + w_1x}}{1 + e^{w_0 + w_1x}}$$

2. **Introduce the substitution $$\( z \):$$**
   Let's define $$\( z = e^{w_0 + w_1x} \)$$. Substituting $$\( z \)$$ into the probability function gives:
   $$p(x) = \frac{z}{1 + z}$$

3. **Calculate \( 1 - p(x) \):**
   To find the expression for $$\( 1 - p(x) \)$$, we subtract $$ p(x)$$ from 1:
   $$1 - p(x) = 1 - \frac{z}{1 + z} = \frac{1 + z - z}{1 + z} = \frac{1}{1 + z}$$

4. **Derive the expression for $$\( \frac{p(x)}{1 - p(x)} \)$$:**
   Now, we form the ratio of $$p(x)$$ to $$\( 1 - p(x) \)$$:
   $$\frac{p(x)}{1 - p(x)} = \frac{\frac{z}{1 + z}}{\frac{1}{1 + z}} = \frac{z}{1 + z} \cdot \frac{1 + z}{1} = z$$

5. **Substitute back for $$\( z \)$$ and conclude:**
   Recalling that $$\( z = e^{w_0 + w_1x} \)$$, we can substitute back to find the odds ratio:
   $$\frac{p(x)}{1 - p(x)} = z = e^{w_0 + w_1x}$$

Thus, we have shown that the odds of \( p(x) \) are given by:
$$\frac{p(x)}{1 - p(x)} = e^{w_0 + w_1x}$$

This result demonstrates how the logistic model relates the log-odds of the dependent variable being 1 to a linear combination of the predictors.




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




