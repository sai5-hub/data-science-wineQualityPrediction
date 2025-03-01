#!/usr/bin/env python
# coding: utf-8

# 
# # Dataset Information
# 
# The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. These datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are munch more normal wines than excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent or poor wines. Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods. 
# 
# Attribute Information:
# Input variables (based on physicochemical tests):
# 1 - fixed acidity
# 2 - volatile acidity
# 3 - citric acid
# 4 - residual sugar
# 5 - chlorides
# 6 - free sulfur dioxide
# 7 - total sulfur dioxide
# 8 - density
# 9 - pH
# 10 - sulphates
# 11 - alcohol
# Output variable (based on sensory data):
# 12 - quality (score between 0 and 10) 
# 

# # Importing the modules

# In[462]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# # Loading the dataset

# In[463]:


# loading the dataset and displaying first 5 rows of the dataset
df = pd.read_csv('winequality.csv')
df.head()


# # Data Analysis

# In[464]:


# number of rows & columns in the dataset
df.shape


# In[465]:


# unique quality values
df["quality"].unique()


# In[466]:


# create count plot for wine type
sns.countplot(df['type'])


# In[467]:


# statistical data
df.describe().T.style.background_gradient(cmap='Greens')


# In[468]:


# datatype information
df.info()


# In[469]:


# check for null values
df.isnull().sum()


# In[470]:


# fill the missing values
for col, value in df.items():
    if col != 'type':
        df[col] = df[col].fillna(df[col].mean())


# In[471]:


# check for null values again
df.isnull().sum()


# In[472]:


# quality count
df['quality'].value_counts()


# In[473]:


# create count plot for quality 
sns.countplot(df['quality'])


# In[474]:


# create box plots
fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    if col != 'type':
        sns.boxplot(y=col, data=df, ax=ax[index])
        index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# In[475]:


# create dist plot
fig, ax = plt.subplots(ncols=6, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    if col != 'type':
        sns.distplot(value, ax=ax[index])
        index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# In[476]:


# log transformation
df['free sulfur dioxide'] = np.log(1 + df['free sulfur dioxide'])
sns.distplot(df['free sulfur dioxide'])


# In[477]:


# constructing a heatmap to understand the correlation between the columns
corr = df.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True)
plt.title('Correlation between the columns')
plt.show()


# In[478]:


df.corr()['quality'].sort_values()


# # Data Processing
# 
# -Replacing the quality values with 0 and 1, where 0 indicates "Bad Quality" and 1 indicates "Good Quality"
# 

# In[479]:


df['quality'] = df.quality.apply(lambda x:1 if x>=6 else 0)


# In[480]:


X = df.drop(columns=['type', 'quality'])
y = df['quality']


# In[481]:


df['quality'].value_counts()


# In[482]:


sns.countplot(df['quality'])


# In[483]:


X = df.drop(columns=['type', 'quality'])
y = df['quality']


# In[484]:


y.value_counts()


# # Train and Test with imbalanced data

# In[485]:


# classify function
def classify(model, X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # train the model
    model.fit(x_train, y_train)
    print("Accuracy:", model.score(x_test, y_test) * 100)
    
    # cross-validation
    score = cross_val_score(model, X, y, cv=5)
    print("CV Score:", np.mean(score)*100)


# # Model Training with imbalanced data

# In[486]:


# Logistic Regression
model = LogisticRegression()
classify(model, X, y)


# In[487]:


metrics.plot_confusion_matrix(model, X, y)
plt.show()


# In[488]:


# Decision Tree
model = DecisionTreeClassifier()
classify(model, X, y)


# In[489]:


metrics.plot_confusion_matrix(model, X, y)
plt.show()


# In[490]:


# Random Forest
model = RandomForestClassifier()
classify(model, X, y)


# In[491]:


metrics.plot_confusion_matrix(model, X, y)
plt.show()


# # Train and Test with balanced data

# In[492]:


oversample = SMOTE(k_neighbors=4)
# transform the dataset
X, y = oversample.fit_resample(X, y)


# In[493]:


y.value_counts()


# In[494]:


# classify function
def classify(model, X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # train the model
    model.fit(x_train, y_train)
    print("Accuracy: ", model.score(x_test, y_test) * 100)
    
    # cross-validation
    score = cross_val_score(model, X, y, cv=5)
    print("CV Score:", np.mean(score)*100)


# # Model Training with balanced data

# In[495]:


# Logistic Regression
model = LogisticRegression()
classify(model, X, y)


# In[496]:


metrics.plot_confusion_matrix(model, X, y)
plt.show()


# In[497]:


# Decision Tree
model = DecisionTreeClassifier()
classify(model, X, y)


# In[498]:


metrics.plot_confusion_matrix(model, X, y)
plt.show()


# In[499]:


# Random Forest
model = RandomForestClassifier()
classify(model, X, y)


# In[500]:


metrics.plot_confusion_matrix(model, X, y)
plt.show()


# # Building a Predictive System

# In[501]:


input_data = (7,0.27,0.36,20.7,0.045,45,170,1.001,3,0.45,8.8)

# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==1):
  print('Good Quality Wine')
else:
  print('Bad Quality Wine')


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




