#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow_probability tensorboard')


# In[3]:


get_ipython().system('pip install tensorflow>=2.3')


# In[4]:


from sklearn.datasets import fetch_openml
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


X, y = fetch_openml(data_id=1565, return_X_y=True, as_frame=True)


# In[6]:


X.head()


# In[7]:


y.astype(int).min()


# In[8]:


y.hist(bins=5)


# In[9]:


target = (y.astype(int) > 1).astype(float)


# In[10]:


from matplotlib import pyplot as plt

target.hist(figsize=(3, 5), rwidth=5)
plt.xticks([0.05, 0.95], ['healthy', 'unhealthy'])
plt.grid(b=None)


# In[11]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_t = scaler.fit_transform(X)


# In[12]:


X_t


# In[13]:


from sklearn.model_selection import train_test_split

Xt_train, Xt_test, y_train, y_test = train_test_split(
    X_t, target, test_size=0.33, random_state=42
)  # for neural networks
X_train, X_test, y_train, y_test = train_test_split(
    X, target, test_size=0.33, random_state=42
)  # for decision tree approaches


# In[14]:


X.isnull().sum(axis=0).any()


# In[15]:


import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow import keras

negloglik = lambda y, p_y: -p_y.log_prob(y)

model = keras.Sequential([
  keras.layers.Dense(12, activation='relu', name='hidden'),
  keras.layers.Dense(1, name='output'),
  tfp.layers.DistributionLambda(
      lambda t: tfd.Bernoulli(logits=t)
  ),
])

model.compile(optimizer=tf.optimizers.Adagrad(learning_rate=0.05), loss=negloglik)


# In[16]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[17]:


callbacks = [
    keras.callbacks.EarlyStopping(patience=10, monitor='loss'),
    keras.callbacks.TensorBoard(log_dir='./logs'),
]
history = model.fit(
    Xt_train,
    y_train.values,
    epochs=10000,
    verbose=False,
    callbacks=callbacks
)


# In[18]:


print(len(history.epoch))


# In[19]:


model.summary()


# In[20]:


y_pred = model(Xt_test)


# In[21]:


def to_one_hot(a):
    """convert from integer encoding to one-hot"""
    b = np.zeros((a.size, 2))
    b[np.arange(a.size), np.rint(a).astype(int)] = 1
    return b


# In[22]:


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "15"


# In[23]:


from scipy.stats import norm
import numpy as np


a, b = y_pred.mean().numpy()[10], y_pred.variance().numpy()[10]
fig, ax = plt.subplots(1, 1)
x = np.linspace(
    norm.ppf(0.001, a, b),
    norm.ppf(0.999, a, b),
    100
)
pdf = norm.pdf(x, a, b)
ax.plot(
    x, 
    pdf / np.sum(pdf), 
    'r-', lw=5, alpha=0.6, 
    label='norm pdf'
)
plt.ylabel('probability density')
plt.xlabel('predictions');


# In[24]:


def to_classprobs(y_pred):
    class_probs = np.zeros(shape=(y_pred.mean().numpy().shape[0], 2))
    for i, (a, b) in enumerate(zip(y_pred.mean().numpy(), y_pred.variance().numpy())):
        conf = norm.cdf(0.5, a, b)
        class_probs[i, 0] = conf
        class_probs[i, 1] = 1 - conf
    return class_probs

class_probs = to_classprobs(y_pred)


# In[25]:


import sklearn


# In[26]:


'auc score: {:.3f}'.format(sklearn.metrics.roc_auc_score(to_one_hot(y_test), class_probs))


# In[27]:


class ModelWrapper(sklearn.base.ClassifierMixin):
    _estimator_type = 'classifier'
    classes_ = [0, 1]
    def predict_proba(self, X):
        pred = model(X)
        return to_classprobs(pred)
    
model_wrapper = ModelWrapper()


# In[28]:


from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score


average_precision = average_precision_score(to_one_hot(y_test), class_probs)
fig = plot_precision_recall_curve(model_wrapper, Xt_test, y_test)
fig.ax_.set_title(
    '2-class Precision-Recall curve: '
    'AP={0:0.2f}'.format(average_precision)
)


# In[29]:


import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow import keras

negloglik = lambda y, p_y: -p_y.log_prob(y)

model = keras.Sequential([
  keras.layers.Dense(12, activation='relu', name='hidden'),
  keras.layers.Dense(1, name='output'),
  tfp.layers.DistributionLambda(
      lambda t: tfd.Bernoulli(logits=t)
  ),
])

model.compile(optimizer=tf.optimizers.Adagrad(learning_rate=0.05), loss=negloglik)

