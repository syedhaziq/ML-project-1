#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[4]:


df=pd.read_csv('news.csv')
df


# In[42]:


df.isna().sum()


# In[9]:


labels=df.label
labels


# In[10]:


X_train,X_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# In[11]:


tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train=tfidf_vectorizer.fit_transform(X_train) 
tfidf_test=tfidf_vectorizer.transform(X_test)


# In[13]:


print(tfidf_train)


# In[14]:


clf=PassiveAggressiveClassifier(max_iter=50)
clf.fit(tfidf_train,y_train)


# In[36]:


y_pred=clf.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(score)


# In[35]:


clf.score(tfidf_test,y_test)


# In[38]:


cf_matrix=confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# In[39]:


import seaborn as sns
sns.heatmap(cf_matrix, annot=True)


# In[40]:


confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# In[ ]:




