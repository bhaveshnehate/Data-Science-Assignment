#!/usr/bin/env python
# coding: utf-8

# # Q1

# In[36]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")


# In[37]:


df = pd.read_csv('delivery_time.csv')
df


# In[38]:


df.describe()


# In[39]:


df.info()


# In[40]:


df.isna().sum()


# # Correlation Analysis

# In[41]:


sns.heatmap(df.corr(), annot=True)


# In[42]:


df.rename({'Delivery Time':'Delivery_Time','Sorting Time':'Sorting_Time'},axis=1,inplace=True)


# In[43]:


plt.scatter(df['Delivery_Time'],df['Sorting_Time'])


# In[44]:


plt.subplot(1,2,1)
sns.distplot(df['Delivery_Time'])
plt.subplot(1,2,2)
sns.distplot(df['Sorting_Time'])


# # Model Building

# In[45]:


model = smf.ols('Delivery_Time~Sorting_Time',data = df).fit()


# In[46]:


model.rsquared,model.rsquared_adj


# In[47]:


model.params


# In[48]:


model.summary()


# In[49]:


new_data = pd.DataFrame({
    'Sorting_Time':[8,4,12,1,9,13,0]
})


# In[50]:


model.predict(new_data)


# In[51]:


new_data['pred_delivary_time'] = model.predict(new_data)
new_data


# In[52]:


plt.subplot(1,2,1)
sns.distplot(np.log(df['Delivery_Time']))
plt.subplot(1,2,2)
sns.distplot(np.log(df['Sorting_Time']))


# In[53]:


model = smf.ols('np.log(Delivery_Time)~np.log(Sorting_Time)',data=df).fit()


# In[54]:


model.summary()


# # Q2

# In[55]:


df = pd.read_csv('Salary_Data.csv')
df


# In[56]:


df.describe()


# In[57]:


df.info()


# In[58]:


df.corr()


# In[59]:


sns.heatmap(df.corr(), annot=True)


# In[60]:


plt.subplot(1,2,1)
sns.distplot(np.log(df['YearsExperience']))
plt.subplot(1,2,2)
sns.distplot(np.log(df['Salary']))


# In[61]:


model = smf.ols('Salary~YearsExperience',data = df).fit()


# In[62]:


c,m = model.params
c,m


# In[63]:


model.rsquared,model.rsquared_adj


# In[64]:


model.summary()


# In[ ]:





# In[65]:


new_data = pd.DataFrame(
{
    'YearsExperience': [2,0,7,10,15,9,4]
})


# In[66]:


model.predict(new_data)


# In[67]:


new_data['pred_salary'] = model.predict(new_data)
new_data


# In[68]:


model = smf.ols('np.sqrt(Salary)~np.sqrt(YearsExperience)',data = df).fit()


# In[69]:


model.rsquared,model.rsquared_adj


# In[70]:


model.summary()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




