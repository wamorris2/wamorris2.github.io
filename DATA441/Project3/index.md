# Project 3
### Willem Morris
---
## Question 1
This implementation of the Gradient Boosting algorithm takes x and y train data, some new data xnew to predict on, and two regressors regressor1 and regressor2 as arguments. It first fits x and y to regressor1 and determines the residuals of its predictions on x. It uses x and these residuals to fit regressor2 and returns the sum of the predictions of regressor1 and regressor2 on xnew.

```python
def gradient_boosted_regression(x, y, xnew, regressor1, regressor2):
  regressor1.fit(x,y)
  residuals1 = y - regressor1.predict(x)
  regressor2.fit(x,residuals1)
  return regressor1.predict(xnew) + regressor2.predict(xnew)
```

---
## Question 2
First I defined the regressors that I would compare:
```python
regressors = [
    [Lowess_AG_MD(f=25/len(x_train), iter=2, intercept=True, kernel=Tricubic), RandomForestRegressor(n_estimators=100,max_depth=3)],
    [Lowess_AG_MD(f=25/len(x_train), iter=2, intercept=True, kernel=Quartic), RandomForestRegressor(n_estimators=100,max_depth=3)],
    [Lowess_AG_MD(f=25/len(x_train), iter=2, intercept=True, kernel=Epanechnikov), RandomForestRegressor(n_estimators=100,max_depth=3)],
    [Lowess_AG_MD(f=25/len(x_train), iter=2, intercept=True, kernel=Gaussian), RandomForestRegressor(n_estimators=100,max_depth=3)],
    [RandomForestRegressor(n_estimators=100,max_depth=3), Lowess_AG_MD(f=25/len(x_train), iter=2, intercept=True, kernel=Tricubic)],
    [RandomForestRegressor(n_estimators=100,max_depth=3), Lowess_AG_MD(f=25/len(x_train), iter=2, intercept=True, kernel=Quartic)],
    [RandomForestRegressor(n_estimators=100,max_depth=3), Lowess_AG_MD(f=25/len(x_train), iter=2, intercept=True, kernel=Epanechnikov)],
    [RandomForestRegressor(n_estimators=100,max_depth=3), Lowess_AG_MD(f=25/len(x_train), iter=2, intercept=True, kernel=Gaussian)],
]
```

### Concrete Data
```python
x = concrete_data.loc[:,'cement':'age'].values
y = concrete_data['strength'].values
x_train, x_test, y_train, y_test = tts(x,y,test_size=0.3,shuffle=True,random_state=123)
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)
for (i, regs) in enumerate(regressors):
  reg1, reg2 = regs[0], regs[1]
  yest = gradient_boosted_regression(x_train, y_train, x_test, reg1, reg2)
  print(i,":",mse(y_test, yest))
``` 
```
output:
0 : 46.859311059019994
1 : 46.14875170966533
2 : 45.65888573476609
3 : 54.26470543368517
4 : 37.98276411607506
5 : 37.070578397320475
6 : 39.82593289740021
7 : 61.65086780689976
```
Observations:
- The lowest MSE was yielded by a boosted gradient of a RandomForestRegressor on the train data and a Lowess regressor with a Quartic weight kernel on the residuals.
- The difference between using a Tricubic and Quartic kernel seems to be marginal.
- Using a RandomForestRegressor first results in lower MSE's except with a Lowess w/ Gaussian kernel regressor.


### Cars Data
```python
x = cars_data.loc[:,'CYL':'WGT'].values
y = cars_data['MPG'].values
x_train, x_test, y_train, y_test = tts(x,y,test_size=0.3,shuffle=True,random_state=123)
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)
for (i, regs) in enumerate(regressors):
  reg1, reg2 = regs[0], regs[1]
  yest = gradient_boosted_regression(x_train, y_train, x_test, reg1, reg2)
  print(i,":",mse(y_test, yest))
```
```
output:
0 : 19.049958461629373
1 : 19.102110830159425
2 : 17.95027857831022
3 : 16.232870297838208
4 : 18.74276193725798
5 : 18.696658826764036
6 : 18.342132795130425
7 : 16.609306581270854
```
Observations:
- A Gaussian Kernel performs the best on both types of models.
- Using Lowess w/ a Gaussian kernel results in the lowest MSE.

### Housing Data
```python
features = ['crime','rooms','residential','industrial','nox','older','distance','highway','tax','ptratio','lstat']
x = housing_data[features].values
y = housing_data['cmedv'].values
x_train, x_test, y_train, y_test = tts(x,y,test_size=0.3,shuffle=True,random_state=123)
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)
for (i, regs) in enumerate(regressors):
  reg1, reg2 = regs[0], regs[1]
  yest = gradient_boosted_regression(x_train, y_train, x_test, reg1, reg2)
  print(i,":",mse(y_test, yest))
```
```
output:
0 : 17.183408862944443
1 : 15.177546025043608
2 : 17.15482507398099
3 : 19.94587400066853
4 : 16.540384689388407
5 : 15.171453482380404
6 : 15.838730786107034
7 : 16.34235626053901
```
Observations:
- The Quartic Kernel produces the lowest possible MSEs when regressing Lowess on both the data and the residuals.
- Regressing the data with Lowess tends to produce higher MSEs.

---
## Question 3
```python
mse_lwr = []
mse_rf = []
kf = KFold(n_splits=10,shuffle=True,random_state=1234)
model_lw = Lowess_AG_MD(f=25/len(xtrain),iter=1,intercept=True, kernel=Tricubic)
model_rf = RandomForestRegressor(n_estimators=200,max_depth=5)

for idxtrain, idxtest in kf.split(x):
  xtrain = x[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = x[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)

  yhat_lw = gradient_boosted_regression(xtrain,ytrain,xtest, model_lw, model_rf)
  
  model_rf.fit(xtrain,ytrain)
  yhat_rf = model_rf.predict(xtest)

  mse_lwr.append(mse(ytest,yhat_lw))
  mse_rf.append(mse(ytest,yhat_rf))
print('The Cross-validated Mean Squared Error for Locally Weighted Regression is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for Random Forest is : '+str(np.mean(mse_rf)))
```
```
output:
The Cross-validated Mean Squared Error for Locally Weighted Regression is : 56.02522991431997
The Cross-validated Mean Squared Error for Random Forest is : 45.60571912050503
```
Random Forest still outpaces gradient boosted Lowess regression in this case. 


## Libraries, Functions, and Classes

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as stats 
from sklearn.model_selection import train_test_split as tts, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error as mse
from scipy.interpolate import interp1d, RegularGridInterpolator, griddata, LinearNDInterpolator, NearestNDInterpolator
from math import ceil
from scipy import linalg
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

scale = StandardScaler()
cars_data = pd.read_csv('drive/MyDrive/Data Sets/cars.csv')
concrete_data = pd.read_csv('drive/MyDrive/Data Sets/concrete.csv')
housing_data = pd.read_csv('drive/MyDrive/Data Sets/housing.csv')

def dist(u,v):
  if len(v.shape)==1:
    v = v.reshape(1,-1) # if v only has one observation, make it a row vector
  d = np.array([np.sqrt(np.sum((u-v[i])**2,axis=1)) for i in range(len(v))])
  return d

# Gaussian Kernel
def Gaussian(x):
  return 1/(np.sqrt(2*np.pi))*np.exp(-1/2*x**2)

# Tricubic Kernel
def Tricubic(x):
  return 70/81*((1-x**3)**3)

# Quartic Kernel
def Quartic(x):
  return 15/16*((1-x**2)**2)

# Epanechnikov Kernel
def Epanechnikov(x):
  return 3/4*((1-x**2)) 

def lw_ag_md(x, y, xnew, kernel=Tricubic, f=2/3,iter=3, intercept=True):

  n = len(x)
  r = int(ceil(f * n))
  yest = np.zeros(n)

  if len(y.shape)==1: # here we make column vectors
    y = y.reshape(-1,1)

  if len(x.shape)==1:
    x = x.reshape(-1,1)
  
  if intercept:
    x1 = np.column_stack([np.ones((len(x),1)),x])
  else:
    x1 = x

  h = [np.sort(np.sqrt(np.sum((x-x[i])**2,axis=1)))[r] for i in range(n)]
  # dist(x,x) is always symmetric
  w = np.clip(dist(x,x) / np.array(h), 0.0, 1.0)
  w = kernel(w)

  #Looping through all X-points
  delta = np.ones(n)
  for iteration in range(iter):
    for i in range(n):
      W = np.diag(delta).dot(np.diag(w[i,:]))
      # when we multiply two diagonal matrices we get also a diagonal matrix
      b = np.transpose(x1).dot(W).dot(y)
      A = np.transpose(x1).dot(W).dot(x1)
      ##
      A = A + 0.0001*np.eye(x1.shape[1]) # if we want L2 regularization for solving the system
      beta = linalg.solve(A, b)

      beta, res, rnk, s = linalg.lstsq(A, b)
      yest[i] = np.dot(x1[i],beta.ravel())

    residuals = y.ravel() - yest
    s = np.median(np.abs(residuals))

    delta = np.clip(residuals / (6.0 * s), -1, 1)

    delta = (1 - delta ** 2) ** 2
    
  # here we are making predictions for xnew by using an interpolation and the predictions we made for the train data
  if x.shape[1]==1:
    f = interp1d(x.flatten(),yest,fill_value='extrapolate')
    output = f(xnew)
  else:
    output = np.zeros(len(xnew))
    for i in range(len(xnew)):
      ind = np.argsort(np.sqrt(np.sum((x-xnew[i])**2,axis=1)))[:r]
      pca = PCA(n_components=3)
      x_pca = pca.fit_transform(x[ind])
      tri = Delaunay(x_pca,qhull_options='QJ Pp')
      f = LinearNDInterpolator(tri,yest[ind])
      output[i] = f(pca.transform(xnew[i].reshape(1,-1))) 
      # the output may have NaN's where the data points from xnew are outside the convex hull of X

  if sum(np.isnan(output))>0:
    g = NearestNDInterpolator(x,yest.ravel()) 
    # output[np.isnan(output)] = g(X[np.isnan(output)])
    output[np.isnan(output)] = g(xnew[np.isnan(output)])
  return output

class Lowess_AG_MD:
    def __init__(self, f = 1/10, iter = 3,intercept=True, kernel=Tricubic):
        self.f = f
        self.iter = iter
        self.intercept = intercept
        self.kernel = kernel
    
    def fit(self, x, y):
        f = self.f
        iter = self.iter
        self.xtrain_ = x
        self.yhat_ = y

    def predict(self, x_new):
        check_is_fitted(self)
        x = self.xtrain_
        y = self.yhat_
        f = self.f
        iter = self.iter
        intercept = self.intercept
        kernel = self.kernel
        return lw_ag_md(x, y, x_new, kernel, f, iter, intercept) # this is actually our defined function of Lowess

    def get_params(self, deep=True):
    # suppose this estimator has parameters "f", "iter" and "intercept"
        return {"f": self.f, "iter": self.iter,"intercept":self.intercept}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
```