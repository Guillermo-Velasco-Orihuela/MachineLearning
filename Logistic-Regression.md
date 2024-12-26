# Logistic Regression

Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model. It is particularly useful for situations where the outcome is binary.

## Fundamental Concepts and Formulas

Logistic regression predicts the probability that a given input belongs to the class labeled "1". The probabilities predicting class "1" are computed with the logistic function, which is defined as:

P(y = 1 | x) = 1 / (1 + exp(-z))

where z is the linear combination of the input features x weighted by the coefficients β derived from the training data:

z = β0 + β1x1 + β2x2 + ... + βnxn

The coefficients are estimated using maximum likelihood estimation (MLE), which aims to find the parameter values that maximize the likelihood of the observed sample.

## Logistic Regression with linear variables

### Implementation

Here's an example of how to implement logistic regression:

##### Libraries

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


INPUTS = ['X1','X2']
OUTPUT = 'Y'
X = df[INPUTS]
y = df[OUTPUT]




