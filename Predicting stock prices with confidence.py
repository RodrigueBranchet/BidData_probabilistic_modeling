#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install yfinance')


# In[2]:


import yfinance as yf

msft = yf.Ticker("MSFT")

msft.info

hist = msft.history(period="max")


# In[3]:


get_ipython().system('pip install -U pymc3')


# In[4]:


get_ipython().system('pip install pymc-learn')


# In[6]:


import os
import theano


# In[7]:


from typing import Tuple
import numpy as np
import pandas as pd
import scipy


def generate_data(
    data: pd.DataFrame, window_size: int, shift: int
) -> Tuple[np.array, np.array]:
    
    y = data.shift(shift + window_size)
    observation_window = []
    for i in range(window_size):
        observation_window.append(
            data.shift(i)
        )
    X = pd.concat(observation_window, axis=1)
    y = (y - X.values[:, -1]) / X.values[:, -1]
    X = X.pct_change(axis=1).values[:, 1:]
    inds = (~np. isnan(X).any(axis=1)) & (~np. isnan(y))
    X, y = X[inds], y[inds]
    return X, y


# In[8]:


from sklearn.model_selection import train_test_split

X, y = generate_data(hist.Close, shift=1, window_size=30)
X_train, X_test, y_train, y_test = train_test_split(X, y)



# In[9]:


pd.DataFrame(X_train).hist();


# In[10]:


from matplotlib import pyplot as plt


# In[11]:


pd.DataFrame(y_train).hist(bins=50)
plt.ylabel('frequency')
plt.xlabel('values')


# In[12]:


(y > 0).mean()


# In[13]:


def threshold_vector(x, threshold=0.02):
    def threshold_scalar(f):
        if f > threshold:
            return 1
        elif f < -threshold:
            return -1
        return 0
    return np.vectorize(threshold_scalar)(x)

y_train_classes, y_test_classes = threshold_vector(y_train), threshold_vector(y_test)
pd.Series(threshold_vector(y)).hist(bins=3)


# In[14]:


import sklearn
import scipy


def to_one_hot(a, classes=[-1, 0, 1]):
    """convert from integer encoding to one-hot"""
    b = np.zeros((a.size, 3))
    b[np.arange(a.size), (np.rint(a)+1).astype(int)] = 1
    return b

def measure_perf(model, y_test):
    y_pred = model.predict(X_test)
    print('AUC: {:.3f}'.format(
        sklearn.metrics.roc_auc_score(
            to_one_hot(y_test), to_one_hot(y_pred), multi_class='ovo'
    )))
    print('mse pred: {}'.format(
        sklearn.metrics.mean_squared_error(y_test, y_pred)
    ))
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)
        y_pred_reg = np.average(
            np.repeat(
                np.array([-1, 0, 1]).reshape(1, -1),
                X_test.shape[0],
                axis=0
            ),
            axis=1,
            weights=y_pred_proba
        )    
        print('mse prob: {}'.format(
            sklearn.metrics.mean_squared_error(y_test, y_pred_reg)
        ))
        

assert to_one_hot(y_test_classes.astype(int)).shape[1] == 3


# In[15]:


pd.Series(y_test).hist();


# In[16]:


pd.DataFrame(to_one_hot(y_train_classes)).hist();


# In[17]:


get_ipython().system('pip3 install imblearn')


# In[18]:


from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.naive_bayes import ComplementNB, CategoricalNB
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import BayesianRidge, LinearRegression, LogisticRegression
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

# see also the naivebayes-predictor library that implements a wider range of supported distribution than what comes with sklearn's implementation

def create_classifier(final_estimator):
    print(f'{final_estimator.__class__.__name__}:')
    if final_estimator._estimator_type == 'regressor':
        estimators = [
            ('rf', RandomForestRegressor(
                n_estimators=100,
                n_jobs=-1
            ))
        ]        
        return StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
        ).fit(X_train, y_train_classes)
    else:
        estimators = [
            ('rf', RandomForestClassifier(
                n_estimators=100,
                n_jobs=-1
            ))
        ]                
        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            #stack_method='predict_proba'
        ).fit(X_train, y_train_classes)

measure_perf(create_classifier(ComplementNB()), y_test_classes)
measure_perf(create_classifier(CategoricalNB()), y_test_classes)
measure_perf(create_classifier(BayesianRidge()), y_test_classes)
measure_perf(create_classifier(LinearRegression()), y_test_classes)


# In[19]:


from sklearn.calibration import CalibratedClassifierCV

rf = RandomForestClassifier(n_estimators=500, n_jobs=-1).fit(X_train, y_train_classes)
platt = CalibratedClassifierCV(rf, method='sigmoid').fit(X_train, y_train_classes)
isotonic = CalibratedClassifierCV(rf, method='isotonic').fit(X_train, y_train_classes)
measure_perf(platt, y_test_classes)
measure_perf(isotonic, y_test_classes)


# In[20]:


measure_perf(rf, y_test_classes)


# In[21]:


from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
# see also the naivebayes-predictor library that implements a wider range of supported distribution than what comes with sklearn's implementation
estimators = [
    ('rf', RandomForestRegressor(n_estimators=500, n_jobs=-1)),
]
clf = StackingRegressor(
    estimators=estimators,
    final_estimator=BayesianRidge()
)
clf.fit(X_train, y_train)
measure_perf(clf, y_test_classes)

