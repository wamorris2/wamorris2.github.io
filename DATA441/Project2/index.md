# Project 2: Alex Gramfort's Locally Weighted Regression
### Willem Morris
---
### Libraries
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import scipy.stats as stats 
from sklearn.model_selection import train_test_split as tts, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error as mse
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from math import ceil
from scipy import linalg
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
```
### Helpful Distance Method
```python
def dist(u,v):
  if len(v.shape)==1:
    v = v.reshape(1,-1) # if v only has one observation, make it a row vector
  d = np.array([np.sqrt(np.sum((u-v[i])**2,axis=1)) for i in range(len(v))])
  return d
```
## **Question 1**
Gramfort's version of Locally Weighted Regression can be modified to take multiple features in the following way:


```python
def lowessag_md(xtrain, ytrain, xtest, f=2/3, iter=3, intercept=True):
  n = len(xtrain)
  r = int(ceil(f * n))
  yest = np.zeros(n)

  # here we make column vectors
  ytrain = ytrain.reshape(-1,1) if len(ytrain.shape)==1 else ytrain
  xtrain = xtrain.reshape(-1,1) if len(xtrain.shape)==1 else xtrain
  
  x1 = np.column_stack([np.ones((n,1)),xtrain]) if intercept else xtrain

  h = [np.sort(np.sqrt(np.sum((xtrain-xtrain[i])**2,axis=1)))[r] for i in range(n)]

  w = np.clip(dist(xtrain,xtrain) / h, 0.0, 1.0)
  w = (1 - w ** 3) ** 3

  #Looping through all X-points
  delta = np.ones(n)
  for iteration in range(iter):
    for i in range(n):
      W = delta * np.diag(w[:,i])
      b = np.transpose(x1).dot(W).dot(ytrain)
      A = np.transpose(x1).dot(W).dot(x1)
      A = A + 0.0001*np.eye(x1.shape[1]) # if we want L2 regularization
      beta = linalg.solve(A, b)
      #beta, res, rnk, s = linalg.lstsq(A, b)
      yest[i] = np.dot(x1[i],beta)

    residuals = ytrain - yest
    s = np.median(np.abs(residuals))
    delta = np.clip(residuals / (6.0 * s), -1, 1)
    delta = (1 - delta ** 2) ** 2

  if x.shape[1]==1:
    f = interp1d(xtrain.flatten(),yest,fill_value='extrapolate')
    output = f(xtest)
  else:
    output = np.zeros(len(xtest))
    for i in range(len(xtest)):
      ind = np.argsort(np.sqrt(np.sum((xtrain-xtest[i])**2,axis=1)))[:r]
      # the following code lets the Delauny triangulation work
      pca = PCA(n_components=3)
      x_pca = pca.fit_transform(xtrain[ind])
      tri = Delaunay(x_pca,qhull_options='QJ')
      f = LinearNDInterpolator(tri,ytrain[ind])
      output[i] = f(pca.transform(xtest[i].reshape(1,-1))) # the output may have NaN's where the data points from xnew are outside the convex hull of X
  if sum(np.isnan(output))>0:
    g = NearestNDInterpolator(xtrain,ytrain.ravel()) 
    output[np.isnan(output)] = g(xtest[np.isnan(output)])
  return output
```

## **Question 2**
This function performs KFold Cross-validation on Lowess regression and returns the mean mse of all of the folds:
```python
def kfold_lw(x, y, n_splits=10, random_state=1234):
  mse_lw = []
  scale = StandardScaler()
  kf = KFold(n_splits=n_splits, shuffle=True, random_state=1234)

  for idxtrain, idxtest in kf.split(x):
    xtrain = scale.fit_transform(x[idxtrain])
    ytrain = y[idxtrain]
    xtest  = scale.transform(x[idxtest])
    ytest  = y[idxtest]

    yest = lowessag_md(xtrain, ytrain, xtest)

    mse_lw.append(mse(ytest, yest))
  
  return np.mean(mse_lw)
```
### Cars Data
```python
data = pd.read_csv('drive/MyDrive/Data Sets/cars.csv')
x = data.loc[:,'CYL':'WGT'].values
y = data['MPG'].values
print('The cross-validated mean squared error for the car data is: '+str(kfold_lw(x,y)))
```
```
Output: The cross-validated mean squared error for the car data is: 21.572708675835393
```

### Concrete Data
```python
data = pd.read_csv('drive/MyDrive/Data Sets/concrete.csv')
x = data.loc[:,'cement':'age'].values
y = data['strength'].values
print('The cross-validated mean squared error for the concrete data is: '+str(kfold_lw(x,y)))
```
```
Output: The cross-validated mean squared error for the concrete data is: 70.38214530730352
```

## Question 3
This is a scikit-learn compliant version of Gramfort's Lowess function:
```python
class Lowess_AG_MD:
    def __init__(self, f = 1/10, iter = 3,intercept=True):
        self.f = f
        self.iter = iter
        self.intercept = intercept
    
    def fit(self, x, y):
        f = self.f
        iter = self.iter
        self.xtrain_ = x
        self.yhat_ = y

    def predict(self, xnew):
        check_is_fitted(self)
        x = self.xtrain_
        y = self.yhat_
        f = self.f
        iter = self.iter
        intercept = self.intercept
        return lowessag_md(x, y, xnew, f, iter, intercept)

    def get_params(self, deep=True):
    # suppose this estimator has parameters "f", "iter" and "intercept"
        return {"f": self.f, "iter": self.iter,"intercept":self.intercept}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
```

Using Grid Search CV and the above class to find the optimal hyperparameters for the cars data:
```python
lwr_pipe = Pipeline([('zscores', StandardScaler()),
                     ('lwr', Lowess_AG_MD())])
params = [{'lwr__f': [1/i for i in range(3,15)],
         'lwr__iter': [1,2,3,4]}]
gs_lowess = GridSearchCV(lwr_pipe,
                      param_grid=params,
                      scoring='neg_mean_squared_error',
                      cv=5)
gs_lowess.fit(x, y)
print(gs_lowess.best_params_)
```
```
Output: {'lwr__f': 0.3333333333333333, 'lwr__iter': 1}
```
Scoring the cars data:
```python
gs_lowess.score(x,y)
```
```
Output: -1.6547449275510313
```