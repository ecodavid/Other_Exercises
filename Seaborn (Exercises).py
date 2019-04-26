#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
pokemon = pd.read_csv('Pokemon (1).csv', index_col = 0)
import seaborn as sns
pokemon.head()


# In[9]:


sns.countplot(pokemon['Generation'])


# In[14]:


sns.kdeplot(pokemon.query('HP').HP)


# In[24]:


sns.distplot(pokemon['HP'], kde = True)


# In[26]:


sns.jointplot(x = 'Defense', y = 'Attack', data = pokemon)


# In[30]:


sns.jointplot(x = 'Defense', y = 'Attack', data = pokemon, kind = 'hex', gridsize = 15)


# In[34]:


sns.kdeplot(pokemon['HP'],pokemon['Attack'])


# In[35]:


df = pokemon[pokemon.Legendary.isin(pokemon.Legendary.value_counts().head().index)]

sns.boxplot(x = 'Legendary', y = 'Attack', data = df)


# In[37]:


sns.violinplot(x = 'Legendary', y = 'Attack', 
    data = pokemon[pokemon.Legendary.isin(pokemon.Legendary.value_counts()[ :5].index)])


# In[ ]:




