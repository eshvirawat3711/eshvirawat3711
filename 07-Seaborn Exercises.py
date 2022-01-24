#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___
# # Seaborn Exercises
# 
# Time to practice your new seaborn skills! Try to recreate the plots below (don't worry about color schemes, just the plot itself.

# ## The Data
# 
# We will be working with a famous titanic data set for these exercises. Later on in the Machine Learning section of the course, we will revisit this data, and use it to predict survival rates of passengers. For now, we'll just focus on the visualization of the data with seaborn:

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


sns.set_style('whitegrid')


# In[3]:


titanic = sns.load_dataset('titanic')


# In[4]:


titanic.head()


# # Exercises
# 
# ** Recreate the plots below using the titanic dataframe. There are very few hints since most of the plots can be done with just one or two lines of code and a hint would basically give away the solution. Keep careful attention to the x and y labels for hints.**
# 
# ** *Note! In order to not lose the plot image, make sure you don't code in the cell that is directly above the plot, there is an extra cell above that one which won't overwrite that plot!* **

# In[17]:





# In[71]:


sns.jointplot(data=titanic,x='fare',y='age')


# In[41]:





# In[44]:





# In[5]:


sns.boxplot(data=titanic, x='class', y='age')


# In[7]:


sns.swarmplot(data=titanic,x='class',y='age',size=3)


# In[46]:





# In[47]:





# In[14]:


sns.heatmap(titanic.corr(),cmap='coolwarm')
plt.title('titanic.corr()')


# In[48]:





# In[21]:


a=sns.FacetGrid(data=titanic,col='sex')
a.map(plt.hist,'age')


# In[49]:





# # Great Job!
# 
# ### That is it for now! We'll see a lot more of seaborn practice problems in the machine learning section!
