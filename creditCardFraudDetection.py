#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


#loading the dataset to Pandas DataFrame
credit_card_data = pd.read_csv("C:/Users/007si/Downloads/creditCardFraud.csv")


# In[3]:


label_encoder = LabelEncoder()
credit_card_data['Merchant'] = label_encoder.fit_transform(credit_card_data['Merchant'])
credit_card_data['Location'] = label_encoder.fit_transform(credit_card_data['Location'])
credit_card_data['Transaction Type'] = label_encoder.fit_transform(credit_card_data['Transaction Type'])


# In[4]:


#first 5 rows 
credit_card_data.head()


# In[5]:


credit_card_data.tail()


# In[6]:


#dataset information
credit_card_data.info()


# In[7]:


#check the number of missing values in each column
credit_card_data.isnull().sum()


# In[8]:


#distribution of legit transaction and fraudlent transactions
# 0 -> legit transaction
# 1 -> fraud transaction
credit_card_data['Fraudulent'].value_counts()


# In[9]:


#this dataset is unbalanced 


# In[10]:


# seperating the data for analysis 
legit = credit_card_data[credit_card_data.Fraudulent == 0]
fraud = credit_card_data[credit_card_data.Fraudulent == 1]


# In[11]:


print(legit.shape)
print(legit.shape)


# In[12]:


#statistical measure of the data
legit.Amount.describe()


# In[13]:


fraud.Amount.describe()


# In[14]:


#compare the values for both transactions
credit_card_data.groupby('Fraudulent').mean()


# In[15]:


#under sampling
#build a sample dataset containing similar distribution of normal transactions and Fraudulent transactions


# In[16]:


legit_sample = legit.sample(n=200)


# In[17]:


#concatnation two dataFrames
new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[18]:


new_dataset.head()


# In[19]:


new_dataset.tail()


# In[20]:


new_dataset['Fraudulent'].value_counts()


# In[21]:


new_dataset.groupby('Fraudulent').mean()


# In[22]:


#Splitting the dataset into Features and Tragets


# In[23]:


X = new_dataset.drop(columns='Fraudulent', axis=1)
Y = new_dataset['Fraudulent']


# In[24]:


print(X)


# In[25]:


print(Y)


# In[26]:


#splitting the data into Training and testing data


# In[27]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[28]:


print(X.shape, X_train.shape, X_test.shape)


# In[29]:


#Model Training using Logistic Regression


# In[30]:


model = LogisticRegression()


# In[31]:


#training the logistic regression model with training data
model.fit(X_train, Y_train)


# In[32]:


#Evaluation of model


# In[33]:


#accuracy Score
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[34]:


print('Accuracy on Training data: ', training_data_accuracy)


# In[35]:


#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[36]:


print("Accuracy on Training data: ", test_data_accuracy)

