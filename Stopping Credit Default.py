#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.datasets import fetch_openml


# In[2]:


openml_frame = fetch_openml(data_id=42477, as_frame=True)


# In[3]:


print(openml_frame['DESCR'])


# In[4]:


data = openml_frame['data']


# In[5]:


data.head()


# In[6]:


data.columns


# In[7]:


openml_frame.keys()


# In[8]:


openml_frame['feature_names']


# In[9]:


data.dtypes


# In[10]:


cat_cols = list(data.select_dtypes(include='category').columns)
num_cols = list(data.select_dtypes(exclude='category').columns)


# In[11]:


cat_cols


# In[12]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
X1 = cat_encoder.fit_transform(data[cat_cols]).todense()
scaler = StandardScaler()
X2 = scaler.fit_transform(
    data[num_cols]
)
X = np.concatenate([X1, X2], axis=1)
feature_names = list(cat_encoder.get_feature_names()) + num_cols


# In[13]:


X.shape


# In[14]:


from sklearn.model_selection import train_test_split

target_dict = {val: num for num, val in enumerate(list(openml_frame['target'].unique()))}
y = openml_frame['target'].apply(lambda x: target_dict[x]).astype('float').values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
tfd = tfp.distributions
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


negloglik = lambda y, rv_y: -rv_y.log_prob(y)


# In[16]:


def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
    tfp.layers.VariableLayer(n, dtype=dtype),
    tfp.layers.DistributionLambda(lambda t: tfd.Independent(
        tfd.Normal(loc=t, scale=1),
        reinterpreted_batch_ndims=1)),
    ])


# In[17]:


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
    tfp.layers.VariableLayer(2 * n, dtype=dtype),
    tfp.layers.DistributionLambda(lambda t: tfd.Independent(
        tfd.Normal(
            loc=t[..., :n],
            scale=1e-5 + tf.nn.softplus(c + t[..., n:])
        ),
    reinterpreted_batch_ndims=1)),
    ])


# In[18]:


model = tf.keras.Sequential([
    tfp.layers.DenseVariational(2, posterior_mean_field, prior_trainable, kl_weight=1/X.shape[0]),
    tfp.layers.DistributionLambda(
        lambda t: tfd.Normal(
            loc=t[..., :1],
            scale=1e-3 + tf.math.softplus(0.01 * t[...,1:])
        )
    ),
])


# In[19]:


model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.01),
    loss=negloglik
)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=1000,
    verbose=False,
    callbacks=[callback]
)


# In[20]:


preds = model(X_test)


# In[21]:




from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, preds.mean().numpy())


# In[22]:


import scipy
scipy.stats.spearmanr(np.abs(y_test - preds.mean().numpy().squeeze()), preds.variance().numpy().squeeze())


# In[23]:


import seaborn as sns
size=25
params = {
    'legend.fontsize': 'large',
    'figure.figsize': (20, 8),
    'axes.labelsize': size,
    'axes.titlesize': size,
    'xtick.labelsize': size * 0.75,
    'ytick.labelsize': size * 0.75,
    'axes.titlepad': 25
}
plt.rcParams.update(params)

error = y_test - preds.mean().numpy().squeeze()
ax = sns.regplot(
    x=preds.variance().numpy().squeeze(),
    y=error*error,
    color='c'
)
plt.ylabel('error')
plt.xlabel('variance');


# In[25]:


from sklearn.metrics import confusion_matrix
import pandas as pd

cm = confusion_matrix(y_test, preds.mean().numpy() >= 0.5)
cm = pd.DataFrame(data=cm / cm.sum(axis=0), columns=['False', 'True'], index=['False', 'True'])
sns.heatmap(
    cm,
    fmt='.2f',
    cmap='Blues',
    annot=True,
    annot_kws={'fontsize': 18}
)

