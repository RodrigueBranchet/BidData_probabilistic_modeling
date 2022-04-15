#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install lifetimes')


# In[2]:


from lifetimes.datasets import load_cdnow_summary, load_cdnow_summary_data_with_monetary_value


# In[3]:


data = load_cdnow_summary_data_with_monetary_value()


# In[4]:


len(data)


# In[5]:


(data.frequency>0).sum()


# In[6]:


data.head()


# In[7]:


data.head()


# In[8]:


data.groupby(by='customer_id').count().mean()


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt


# In[10]:


fig = data.frequency.hist(bins=30)
plt.ylabel('customers')
plt.xlabel('number of purchases')
plt.xlim(left=0, right=20)


# In[11]:


data.frequency.mean()


# In[12]:


data.mean()


# In[15]:


get_ipython().system('pip install dython')


# In[16]:


from dython.nominal import associations
associations(data)


# In[17]:


import scipy

scipy.stats.spearmanr(data.recency, data.monetary_value)


# In[18]:


scipy.stats.spearmanr(data.recency, data.monetary_value * data.frequency)


# In[19]:


scipy.stats.spearmanr(data.recency, data.monetary_value)


# In[21]:


from lifetimes import GammaGammaFitter, BetaGeoFitter
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(data['frequency'], data['recency'], data['T'])


# In[22]:


from lifetimes.plotting import plot_frequency_recency_matrix

plot_frequency_recency_matrix(bgf)


# In[23]:


from lifetimes.plotting import plot_probability_alive_matrix

plot_probability_alive_matrix(bgf)


# In[24]:


data_repeat = data[data.frequency>0]


# In[25]:


ggf = GammaGammaFitter(penalizer_coef=0.0)
ggf.fit(data_repeat.frequency, data_repeat.monetary_value)


# In[26]:


print(ggf.conditional_expected_average_profit(
        data['frequency'],
        data['monetary_value']
    ).head(10))


# In[27]:


print('Expected conditional average profit: %s, Average profit: %s' % (
    ggf.conditional_expected_average_profit(
        data['frequency'],
        data['monetary_value']
    ).mean(),
    data[data['frequency']>0]['monetary_value'].mean()
))


# In[28]:


print(ggf.customer_lifetime_value(
    bgf, 
    data['frequency'],
    data['recency'],
    data['T'],
    data['monetary_value'],
    time=12, 
    discount_rate=0.01 
).head(10))

