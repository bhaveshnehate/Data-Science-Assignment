#!/usr/bin/env python
# coding: utf-8

# # Assignment No.3

# # Q1- Hypothesis Testing Exercise

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
import scipy


# In[2]:


df = pd.read_csv(r"G:\DS ASS\03 Hypothesis Testing\Cutlets.csv")
df


# In[3]:


s,p = scipy.stats.ttest_ind(df['Unit A'],df['Unit B'])
s,p


# In[4]:


#Compare P_value with alph=0.5


# In[5]:


alph = 0.05
if p<=alph:
    print('reject null hypothesis')
else:
    print('fail to reject null hypothesis')


# ##### i.e Accept null hypothesis
# their is no difference between unit A and unit B

# In[ ]:





# #### Q2 --ANOVA TEST OR F1 TEST
# 

# In[6]:


df = pd.read_csv('LabTAT.csv')
df


# ###### Ho- Their is no differnce in report of TAT
# Ha- Thier is diffence in report of TAT

# In[7]:


s,p = scipy.stats.f_oneway(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],df.iloc[:,3])
s,p


# In[8]:


alph = 0.05
if p<=alph:
    print('reject null hypothesis')
else:
    print('fail to reject null hypothesis')


# ##### Reject null hyp mean accept alt hyp
# thier is difference in average TAT among the different laboratories

# In[ ]:





# # Q3. Two sample t test
# 

# In[9]:


df = pd.read_csv('BuyerRatio.csv',).T
df


# In[10]:


data = {'obs value': ['East', 'West', 'North', 'South'],
        'Male': [50, 142, 131, 70],
        'Female': [435, 1523, 1365, 750]}

df = pd.DataFrame(data)
df


# In[11]:


contingency_table = pd.crosstab(df['obs value'], [df['Male'], df['Female']])


# In[12]:


chi2 = scipy.stats.chi2_contingency(contingency_table)
chi2


# In[13]:


alph = 0.05
if p<=alph:
    print('reject null hypothesis')
else:
    print('fail to reject null hypothesis')


# ###### Accept null hypothesis
# That means the male and felmale BuyerRatio is same

# In[ ]:





# ###### Q4 Chi square

# In[14]:


df = pd.read_csv('Costomer+OrderForm.csv')
df


# In[15]:


df['Phillippines'].value_counts()


# In[16]:


df['Indonesia'].value_counts()


# In[17]:


df['Malta'].value_counts()


# In[18]:


df['India'].value_counts()


# In[19]:


valuecount = np.array([[271,267,269,280],[29,33,31,20]])
valuecount


# In[20]:


chi2,p,dof,exp=scipy.stats.chi2_contingency(valuecount)


# In[21]:


chi2,p,dof,exp


# In[22]:


alph = 0.05
if p<=alph:
    print('reject null hypothesis')
else:
    print('fail to reject null hypothesis')


# # fail to reject null hypothesis
# there is no significant difference in the defective percentage across the four country.

# In[ ]:





# In[ ]:




