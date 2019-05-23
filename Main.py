
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Binarizer


# In[2]:


d = pd.read_csv('bank-additional-full.csv',delimiter=';')


# In[3]:


print(d.shape)


# In[4]:


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# In[5]:


print(d.iloc[0])


# In[6]:


c = MultiColumnLabelEncoder(columns = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y']).fit_transform(d)
corrmat = c.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = 0.8, square = True,cmap='coolwarm')
plt.show()


# In[7]:


c.drop(c.columns[[0,1,2,3,4,5,6,7,8,9]], axis=1, inplace=True)


# In[8]:


corrmat = c.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = 0.8, square = True,cmap = "coolwarm")
plt.show()


# In[9]:


from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression


# In[10]:


train,test = train_test_split(c,test_size = 0.1)
columns = c.columns.tolist()
columns = [c for c in columns if c not in ["y"]]
target = "y"
x_train=train[columns]
y_train=train[target]
print(x_train.shape)
print(y_train.shape)

columns = c.columns.tolist()
columns = [c for c in columns if c not in ["y"]]
target = "y"
x_test=test[columns]
y_test=test[target]
print(x_test.shape)
print(y_test.shape)


# In[ ]:


state = 1
classifiers = {
    "AdaBoostClassifier": AdaBoostClassifier(),
    "LogisticRegression":LogisticRegression(),
    "BaggingClassifier": BaggingClassifier(),
    "ExtraTreesClassifier": ExtraTreesClassifier(),
    "RandomForestClassifier": RandomForestClassifier(bootstrap=False,random_state=1),
    "SVC": SVC(kernel='poly', verbose=True)
}

for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    # fit the data and tag outliers
    if clf_name == "AdaBoostClassifier":
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        print(clf_name+" Accuracy: " + str(round(accuracy_score(y_pred,y_test)*100,2) )+"%")        

    elif clf_name == "BaggingClassifier":
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        print(clf_name+" Accuracy: " + str(round(accuracy_score(y_pred,y_test)*100,2) )+"%")
        
    elif clf_name == "DecisionTreeClassifier":
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        print(clf_name+" Accuracy: " + str(round(accuracy_score(y_pred,y_test)*100,2) )+"%")
        
    elif clf_name == "LogisticRegression":
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        print(clf_name+" Accuracy: " + str(round(accuracy_score(y_pred,y_test)*100,2) )+"%")

    elif clf_name == "IsolationForest":
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        print(clf_name+" Accuracy: " + str(round(accuracy_score(y_pred,y_test)*100,2) )+"%")        

    elif clf_name == "ExtraTreesClassifier":
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        print(clf_name+" Accuracy: " + str(round(accuracy_score(y_pred,y_test)*100,2) )+"%")        

    elif clf_name == "RandomForestClassifier":
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        print(clf_name+" Accuracy: " + str(round(accuracy_score(y_pred,y_test)*100,2) )+"%")        

    elif clf_name == "SVC":
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        print(clf_name+" Accuracy: " + str(round(accuracy_score(y_pred,y_test)*100,2) )+"%") 

