#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
import seaborn as sn
from scipy import stats

import warnings
warnings.filterwarnings('ignore')


# # question 7

# In[18]:


q7 = pd.read_csv(r"G:\DS ASS\01 Basic Statistics_Level 1\Q7.csv")
q7


# In[21]:


q7df=q7[["Points","Score","Weigh"]]
q7df


# In[22]:


q7df.describe()


# In[5]:


q7['Points'].median()


# In[ ]:





# In[ ]:





# In[34]:


stats.mode(q7df['Points'])


# In[35]:


stats.mode(q7df['Score'])


# In[36]:


stats.mode(q7df['Weigh'])


# In[37]:


q7df.var()


# In[38]:


q7.rename(columns={'Unnamed: 0':'Cars'}, inplace  = True)


# In[39]:


q7


# In[48]:


plt.hist(q7["Points"], bins = 10, edgecolor= 'black')
plt.show()


# In[49]:


plt.boxplot(x = 'Points', data =q7)
plt.xlabel('Points')
plt.ylabel('Density')
plt.savefig("PointsInferences.png")
plt.show()


# In[50]:


plt.hist(q7["Score"], bins = 20, edgecolor = 'y')
plt.show()


# In[51]:


plt.boxplot(x = 'Score', data= q7)
plt.xlabel('Scores')
plt.ylabel('Density')
plt.savefig("ScoresInferences.png")
plt.show()


# In[52]:


plt.hist(q7["Weigh"], bins=20, edgecolor = 'red')
plt.show()


# In[53]:


plt.boxplot(x= "Weigh", data = q7)
plt.xlabel('Weigh')
plt.ylabel('Density')
plt.savefig("WeighInferences.png")
plt.show()


# In[54]:


plt.figure(figsize=(16,9))
plt.barh(q7["Cars"], q7["Points"])
plt.yticks(fontsize=14)
plt.show()


# In[55]:


plt.figure(figsize=(16,9))
plt.barh(q7["Cars"], q7["Score"])
plt.yticks(fontsize=14)
plt.show()


# In[56]:


plt.figure(figsize=(16,9))
plt.barh(q7["Cars"], q7["Weigh"])
plt.yticks(fontsize=14)
plt.show()


# # Inferences:
# 
# a) For Points dataset:
# 1) The data is concentrated aroound Median
# 2) There are no outliars
# 3) The distribution is Right skewed
# 
# b) For Score dataset:
# 1) The data is concentrated around Median
# 2) There are 3 Outliars: 5.250, 5.424, 5.345
# 3) The distribution is Left skewed
# 
# c) For Weigh dataset:
# 1) The data is concentrated around Median
# 2) There is 1 Outliar: 22.90
# 3) The distribution is Left skewed

# In[ ]:





# # question 6

# In[58]:


def expected_value(values, weights):
    values = np.asarray(values)
    weights = np.asarray(weights)
    return (values * weights).sum() / weights.sum()


# In[64]:


c_count = [1,4,3,5,6,2]
ch_prob = [0.015,0.20,0.65,0.005,0.01,0.120]
expected_value(c_count, ch_prob)


# # question 8

# In[65]:


weigh = [108,110,123,134,135,145,167,187,199]
probs = [1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9]
expected_value(weigh, probs)


# In[66]:


ch = 1/9
ch


# # question 11

# In[67]:


from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[71]:


conf_z_94 = stats.norm.interval(0.94, loc = 200, scale = 30/np.sqrt(2000))
np.round(conf_z_94,0)


# In[72]:


conf_z_96 = stats.norm.interval(0.96, loc = 200, scale = 30/np.sqrt(2000))
np.round(conf_z_94,0)


# In[73]:


conf_z_98 =  stats.norm.interval(0.98, loc=200,scale=30/np.sqrt(2000))
np.round(conf_z_98,0)


# In[ ]:





# In[74]:


stats.t.ppf(0.03,df=1999)


# In[75]:


stats.t.ppf(0.01,df=1999)


# In[76]:


stats.t.ppf(0.02,df=1999)


# # question 12

# In[77]:


q12 = [34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56]


# In[78]:


stat.mean(q12)


# In[79]:


stat.median(q12)


# In[80]:


stat.variance(q12)


# In[81]:


stat.stdev(q12)


# In[82]:


q12_df = pd.DataFrame({'students':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
                    'marks':(q12)})
q12_df


# In[83]:


q12_df.set_index('students')


# # Question 24

# In[84]:


x_bar = 260
pop_mean = 270


# In[85]:


t_value = (260-270)/(90/np.sqrt(18))
t_value


# In[86]:


1-stats.t.cdf(abs(t_value),df = 17)


# # question 20

# In[88]:


q20 = pd.read_csv(r"G:\DS ASS\01 Basic Statistics_Level 1\Cars.csv")
q20


# In[89]:


from scipy import stats


# In[90]:


q20.describe()


# In[93]:


Prob_MPG_greater_than_38 = np.round(1 - stats.norm.cdf(38, loc= q20.MPG.mean(), scale= q20.MPG.std()),3)
print('P(MPG>38)=',Prob_MPG_greater_than_38)


# In[94]:


prob_MPG_less_than_40 = np.round(stats.norm.cdf(40, loc = q20.MPG.mean(), scale = q20.MPG.std()),3)
print('P(MPG<40)=',prob_MPG_less_than_40)


# In[95]:


prob_MPG_greater_than_20 = np.round(1-stats.norm.cdf(20, loc = q20.MPG.mean(), scale = q20.MPG.std()),3)
print('p(MPG>20)=',(prob_MPG_greater_than_20))


# In[96]:


prob_MPG_less_than_50 = np.round(stats.norm.cdf(50, loc = q20.MPG.mean(), scale = q20.MPG.std()),3)
print('P(MPG<50)=',(prob_MPG_less_than_50))


# In[97]:


prob_MPG_greaterthan20_and_lessthan50= (prob_MPG_less_than_50) - (prob_MPG_greater_than_20)
print('P(20<MPG<50)=',(prob_MPG_greaterthan20_and_lessthan50))


# # Question 22

# In[98]:


# z value for 90% confidence interval
print('Z score for 60% Conifidence Intervla =',np.round(stats.norm.ppf(.05),4))


# In[99]:


# z value for 94% confidence interval
print('Z score for 60% Conifidence Intervla =',np.round(stats.norm.ppf(.03),4))


# In[100]:


# z value for 60% confidence interval
print('Z score for 60% Conifidence Intervla =',np.round(stats.norm.ppf(.2),4))


# In[101]:


# t score for 95% confidence interval
print('T score for 95% Confidence Interval =',np.round(stats.t.ppf(0.025,df=24),4))


# In[102]:


# t value for 94% confidence interval
print('T score for 94% Confidence Inteval =',np.round(stats.t.ppf(0.03,df=24),4))


# In[103]:


# t value for 99% Confidence Interval
print('T score for 95% Confidence Interval =',np.round(stats.t.ppf(0.005,df=24),4))


# # Question 9

# In[105]:


q9a = pd.read_csv(r"G:\DS ASS\01 Basic Statistics_Level 1\Q9_a.csv", index_col = 'Index')
q9a


# In[106]:


print('For Cars Speed', "Skewness value=", np.round(q9a.speed.skew(),2), 'and' , 'Kurtosis value=', np.round(q9a.speed.kurt(),2))


# In[107]:


print('Skewness value =', np.round(q9a.dist.skew(),2),'and', 'Kurtosis value =', np.round(q9a.dist.kurt(),2), 'for Cars Distance')


# In[110]:


q9b =pd.read_csv(r"G:\DS ASS\01 Basic Statistics_Level 1\Q9_b.csv")
q9b


# In[111]:


q9b.rename(columns = {'Unnamed: 0':'Index'}, inplace = True)
q9b


# In[112]:


q9b


# In[126]:


print('For SP Skewness =', np.round(q9b.SP.skew(),2), 'kurtosis =', np.round(q9b.SP.kurt(),2))


# In[127]:


print('For WT Skewness =', np.round(q9b.WT.skew(),2), 'Kurtosis =', np.round(q9b.WT.kurt(),2))


# # Question 21

# In[128]:


q21a = pd.read_csv(r"G:\DS ASS\01 Basic Statistics_Level 1\Cars.csv")
q21a


# In[129]:


import numpy as np
import matplotlib.pyplot as plt

mean, cov = [0, 0], [(1, .6), (.6, 1)]
x, y = np.random.multivariate_normal(mean, cov, 100).T
y += x + 1

f, ax = plt.subplots(figsize=(6, 6))

ax.scatter(x, y, c=".3")
ax.set(xlim=(-3, 3), ylim=(-3, 3))

# Plot your initial diagonal line based on the starting
# xlims and ylims.
diag_line, = ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")

def on_change(axes):
    # When this function is called it checks the current
    # values of xlim and ylim and modifies diag_line
    # accordingly.
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()
    diag_line.set_data(x_lims, y_lims)

# Connect two callbacks to your axis instance.
# These will call the function "on_change" whenever
# xlim or ylim is changed.
ax.callbacks.connect('xlim_changed', on_change)
ax.callbacks.connect('ylim_changed', on_change)

plt.show()


# In[130]:


plt.hist(q21a["MPG"], bins = 20, edgecolor=  'black')
plt.show()


# In[131]:


plt.boxplot(x= 'MPG', data =q21a)
plt.show()


# In[132]:


import statsmodels.api as sm
sm.qqplot(q21a['MPG'])
plt.xlabel('MPG', color ='red')
plt.savefig('MPG of cars.png')
plt.show()


# In[133]:


import scipy.stats as stats
stats.probplot(q21a['MPG'], dist="norm", plot=plt)
plt.xlabel('MPG', color ='red')
plt.savefig('MPG of cars.png')
plt.show()


# In[134]:


sn.distplot(q21a['MPG'],kde=True, bins =10)
plt.show()


# In[135]:


q21b = pd.read_csv(r"G:\DS ASS\01 Basic Statistics_Level 1\wc-at.csv")
q21b


# In[136]:


plt.hist(q21b['Waist'], edgecolor= 'red')
plt.show()


# In[137]:


plt.boxplot(x = 'Waist', data= q21b)
plt.title("Waist")
plt.savefig('Waist.png')
plt.show()


# In[138]:


sn.distplot(q21b['Waist'], 
             bins=10,
            kde = True
            )
plt.show()


# In[139]:


import statsmodels.api as sm
sm.qqplot(q21b['Waist'])
plt.show()


# In[140]:


stats.probplot(q21b['Waist'], dist = 'norm', plot = plt)
plt.xlabel('Waist', color= 'red')
plt.savefig('Waist.png')
plt.show()


# In[141]:


sn.distplot(q21b['AT'], bins =10, kde=True)
plt.show()


# In[142]:


import statsmodels.api as sm
sm.qqplot(q21b['AT'])
plt.show()


# In[143]:


stats.probplot(q21b['AT'], dist = 'norm', plot = plt)
plt.xlabel('AT', color= 'red')
plt.savefig('AT.png')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




