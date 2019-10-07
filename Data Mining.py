#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


fifa=pd.read_csv(r'C:\Users\amshu\Downloads\data.csv')
fifa.head(5)


# In[4]:


fifa.info()


# In[5]:


fifa.isnull().values.any()


# In[6]:


useful_feat     = ['Name',
                   'Age',
                   'Overall',
                   'Potential', 
                   'Value',
                   'Wage',
                   'Preferred Foot',
                   'International Reputation',
                   'Weak Foot',
                   'Skill Moves',
                   'Work Rate',
                   'Body Type',
                   'Position',
                   'Height',
                   'Weight',
                   'Crossing', 
                   'Finishing',
                   'HeadingAccuracy',
                   'ShortPassing', 
                   'Volleys', 
                   'Dribbling',
                   'Curve',
                   'FKAccuracy',
                   'LongPassing',
                   'BallControl',
                   'Acceleration',
                    'SprintSpeed',
                   'Agility',
                   'Reactions', 
                   'Balance',
                   'ShotPower', 
                   'Jumping',
                   'Stamina', 
                   'Strength',
                   'LongShots',
                   'Aggression',
                   'Interceptions',
                   'Positioning', 
                   'Vision', 
                   'Penalties',
                   'Composure',
                   'Marking',
                   'StandingTackle', 
                   'SlidingTackle',
                   'GKDiving',
                   'GKHandling',
                   'GKKicking',
                   'GKPositioning',
                   'GKReflexes']


# In[7]:


fifa1=pd.DataFrame(fifa,columns=useful_feat)
fifa1.head(5)


# In[8]:


fifa1=fifa1.dropna()
fifa1.isnull().any()


# In[9]:


fifa1.isnull().values.any()


# In[10]:


plt.figure(1, figsize=(18, 7))
sns.countplot( x= 'Age', data=fifa, palette='Accent')
plt.title('Age distribution of all players')
plt.show()


# In[11]:


fifa1.describe()


# In[12]:


ml_cols =          ['Crossing', 
                   'Finishing',
                   'HeadingAccuracy',
                   'ShortPassing', 
                   'Volleys', 
                   'Dribbling',
                   'Curve',
                   'FKAccuracy',
                   'LongPassing',
                   'BallControl',
                   'Acceleration',
                    'SprintSpeed',
                   'Agility',
                   'Reactions', 
                   'Balance',
                   'ShotPower', 
                   'Jumping',
                   'Stamina', 
                   'Strength',
                   'LongShots',
                   'Aggression',
                   'Interceptions',
                   'Positioning', 
                   'Vision', 
                   'Penalties',
                   'Composure',
                   'Marking',
                   'StandingTackle', 
                   'SlidingTackle',
                    'GKDiving',
                   'GKHandling',
                   'GKKicking',
                   'GKPositioning',
                   'GKReflexes',
                   'Overall']


# In[13]:


df_ml = pd.DataFrame(data=fifa1, columns=ml_cols)
df_ml.head(10)


# In[14]:


df_ml.describe()


# In[78]:


plt.figure(1, figsize=(40, 25))
mean_val=df_ml.mean()
mean_val.plot(kind='bar')
plt.xlabel('Attributes', fontdict=None, labelpad=None,fontsize=36)
plt.ylabel('Mean', fontdict=None, labelpad=None,fontsize=36)
plt.rc('xtick', labelsize=24)
plt.rc('ytick', labelsize=24) 
plt.show()


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[17]:



df_ml=df_ml.dropna()
df_ml.isnull().any()


# In[18]:


y = df_ml['Overall']
X = df_ml[['Crossing', 
           'Finishing',
           'HeadingAccuracy',
           'ShortPassing', 
           'Volleys', 
           'Dribbling',
           'Curve',
           'FKAccuracy',
           'LongPassing',
           'BallControl',
           'Acceleration',
            'SprintSpeed',
           'Agility',
           'Reactions', 
           'Balance',
           'ShotPower', 
           'Jumping',
           'Stamina', 
           'Strength',
           'LongShots',
           'Aggression',
           'Interceptions',
           'Positioning', 
           'Vision', 
           'Penalties',
           'Composure',
           'Marking',
           'StandingTackle', 
           'SlidingTackle',
           'GKDiving',
            'GKHandling',
            'GKKicking',
            'GKPositioning',
            'GKReflexes']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=101)


# In[19]:


lm = LinearRegression()


# In[20]:



lm.fit(X_train, y_train)


# In[21]:


print('Coefficients:', lm.coef_)


# In[22]:




predictions = lm.predict(X_test)


# In[23]:




plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted y')
plt.show()


# In[24]:


coeffecients = pd.DataFrame(lm.coef_, X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# In[25]:


vals = ['RF', 'ST', 'LW', 'RCM', 'LF', 'RS', 'RCB', 'LCM', 'CB',
       'LDM', 'CAM', 'CDM', 'LS', 'LCB', 'RM', 'LAM', 'LM', 'LB', 'RDM',
       'RW', 'CM', 'RB', 'RAM', 'CF', 'RWB', 'LWB']
ml_players= fifa1.loc[fifa1['Position'].isin(vals) & fifa1['Position']]


# In[26]:


ml_cols1 =          ['Crossing', 
                   'Finishing',
                   'HeadingAccuracy',
                   'ShortPassing', 
                   'Volleys', 
                   'Dribbling',
                   'Curve',
                   'FKAccuracy',
                   'LongPassing',
                   'BallControl',
                   'Acceleration',
                    'SprintSpeed',
                   'Agility',
                   'Reactions', 
                   'Balance',
                   'ShotPower', 
                   'Jumping',
                   'Stamina', 
                   'Strength',
                   'LongShots',
                   'Aggression',
                   'Interceptions',
                   'Positioning', 
                   'Vision', 
                   'Penalties',
                   'Composure',
                   'Marking',
                   'StandingTackle', 
                   'SlidingTackle',
                    'Overall'
                   ]


# In[27]:


dfml = pd.DataFrame(data=ml_players, columns=ml_cols1)
dfml.head(5)


# In[28]:


dfml.count()


# In[29]:


y1 = dfml['Overall']
X1 = dfml[['Crossing', 
           'Finishing',
           'HeadingAccuracy',
           'ShortPassing', 
           'Volleys', 
           'Dribbling',
           'Curve',
           'FKAccuracy',
           'LongPassing',
           'BallControl',
           'Acceleration',
            'SprintSpeed',
           'Agility',
           'Reactions', 
           'Balance',
           'ShotPower', 
           'Jumping',
           'Stamina', 
           'Strength',
           'LongShots',
           'Aggression',
           'Interceptions',
           'Positioning', 
           'Vision', 
           'Penalties',
           'Composure',
           'Marking',
           'StandingTackle', 
           'SlidingTackle']]
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.2, random_state=101)


# In[30]:


lm1 = LinearRegression()


# In[31]:




lm.fit(X_train1, y_train1)


# In[32]:


print('Coefficients:', lm.coef_)


# In[33]:


predictions = lm.predict(X_test1)


# In[34]:


coeffecients = pd.DataFrame(lm.coef_, X1.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# In[35]:


cols =          [   'Overall',
                    'Position',                 
                    'Crossing', 
                   'Finishing',
                   'HeadingAccuracy',
                   'ShortPassing', 
                   'Volleys', 
                   'Dribbling',
                   'Curve',
                   'FKAccuracy',
                   'LongPassing',
                   'BallControl',
                   'Acceleration',
                    'SprintSpeed',
                   'Agility',
                   'Reactions', 
                   'Balance',
                   'ShotPower', 
                   'Jumping',
                   'Stamina', 
                   'Strength',
                   'LongShots',
                   'Aggression',
                   'Interceptions',
                   'Positioning', 
                   'Vision', 
                   'Penalties',
                   'Composure',
                   'Marking',
                   'StandingTackle', 
                   'SlidingTackle',
                    'GKDiving',
                   'GKHandling',
                   'GKKicking',
                   'GKPositioning',
                   'GKReflexes',
                   ]


# In[36]:


cols_no_gk=    [   'Overall',
                    'Position',                 
                    'Crossing', 
                   'Finishing',
                   'HeadingAccuracy',
                   'ShortPassing', 
                   'Volleys', 
                   'Dribbling',
                   'Curve',
                   'FKAccuracy',
                   'LongPassing',
                   'BallControl',
                   'Acceleration',
                    'SprintSpeed',
                   'Agility',
                   'Reactions', 
                   'Balance',
                   'ShotPower', 
                   'Jumping',
                   'Stamina', 
                   'Strength',
                   'LongShots',
                   'Aggression',
                   'Interceptions',
                   'Positioning', 
                   'Vision', 
                   'Penalties',
                   'Composure',
                   'Marking',
                   'StandingTackle', 
                   'SlidingTackle'
                   ]


# In[39]:


df_ml1 = pd.DataFrame(data=fifa1, columns=cols)
df_ml1.head(10)


# In[40]:


df_ml1=df_ml1.replace(to_replace= 'RF',value='RW')
df_ml1=df_ml1.replace(to_replace= 'LF',value='LW')
df_ml1=df_ml1.replace(to_replace= 'LS',value='ST')
df_ml1=df_ml1.replace(to_replace= 'RS',value='ST')
df_ml1=df_ml1.replace(to_replace= 'RAM',value='CAM')
df_ml1=df_ml1.replace(to_replace= 'LAM',value='CAM')
df_ml1=df_ml1.replace(to_replace= 'RCM',value='CM')
df_ml1=df_ml1.replace(to_replace= 'LCM',value='CM')
df_ml1=df_ml1.replace(to_replace= 'RDM',value='CDM')
df_ml1=df_ml1.replace(to_replace= 'LDM',value='CDM')
df_ml1=df_ml1.replace(to_replace= 'RCB',value='CB')
df_ml1=df_ml1.replace(to_replace= 'LCB',value='CB')
df_ml1=df_ml1.replace(to_replace= 'RWB',value='RB')
df_ml1=df_ml1.replace(to_replace= 'LWB',value='LB')
df_ml1=df_ml1.replace(to_replace= 'CF',value='ST')
print(df_ml1)


# In[41]:


df_ml1['Position'].unique()


# In[42]:


df_ml1['Position'].isnull().values.any()


# In[43]:


df_ml1.isnull().values.sum()


# In[44]:


df_ml1.shape


# df_ml1.shape()

# In[45]:


df_ml1.info()


# In[47]:


vals = ['RF', 'ST', 'LW', 'RCM', 'LF', 'RS', 'RCB', 'LCM', 'CB',
       'LDM', 'CAM', 'CDM', 'LS', 'LCB', 'RM', 'LAM', 'LM', 'LB', 'RDM',
       'RW', 'CM', 'RB', 'RAM', 'CF', 'RWB', 'LWB']
ml1= df_ml1.loc[df_ml1['Position'].isin(vals) & df_ml1['Position']]


# In[48]:


df_ml2=df_ml1.select_dtypes(include=[np.number])
df_norm = (df_ml2 - df_ml2.min()) / (df_ml2.max() - df_ml2.min())
df_ml1[df_norm.columns]=df_norm
print(df_ml1)


# In[49]:


ml2=ml1.select_dtypes(include=[np.number])
norm = (ml2 - ml2.min()) / (ml2.max() - ml2.min())
ml1[norm.columns]=norm
print(ml1)


# In[52]:



mapping_all = {'ST': 0, 'RW': 1, 'LW': 2, 'RM': 3, 'CM': 4, 'LM': 5, 'CAM': 6, 'CDM': 7, 'CB': 8, 'LB': 9, 'RB': 10,'GK':11}
df_ml3 = df_ml1.replace({'Position': mapping_all})
print(df_ml3)


# In[53]:



mapping_all1 = {'ST': 0, 'RW': 1, 'LW': 2, 'RM': 3, 'CM': 4, 'LM': 5, 'CAM': 6, 'CDM': 7, 'CB': 8, 'LB': 9, 'RB': 10,}
ml3 = ml1.replace({'Position': mapping_all1})
print(ml3)


# In[54]:


y2 = df_ml3['Position']
X2 = df_ml3[['Crossing', 
           'Finishing',
           'HeadingAccuracy',
           'ShortPassing', 
           'Volleys', 
           'Dribbling',
           'Curve',
           'FKAccuracy',
           'LongPassing',
           'BallControl',
           'Acceleration',
            'SprintSpeed',
           'Agility',
           'Reactions', 
           'Balance',
           'ShotPower', 
           'Jumping',
           'Stamina', 
           'Strength',
           'LongShots',
           'Aggression',
           'Interceptions',
           'Positioning', 
           'Vision', 
           'Penalties',
           'Composure',
           'Marking',
           'StandingTackle', 
           'SlidingTackle',
             'GKDiving',
            'GKHandling',
            'GKKicking',
            'GKPositioning',
            'GKReflexes',]]

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split( X2,y2, random_state=34)


# In[55]:


from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier


# In[56]:


print('X train shape: {}'.format(X_train_all.shape))
print('X test shape: {}'.format(X_test_all.shape))
print('y train shape: {}'.format(y_train_all.shape))
print('y test shape: {}'.format(y_test_all.shape))


# In[57]:




clf_d = DummyClassifier(strategy = 'most_frequent').fit(X_train_all, y_train_all)
acc_d = clf_d.score(X_test_all, y_test_all)
print ('Dummy Classifier (most frequent class): {}'.format(acc_d))



# In[58]:


clf = LogisticRegression().fit(X_train_all, y_train_all)
acc = clf.score(X_test_all, y_test_all)
print ('Logistic Regression Accuracy: {}'.format(acc))


# In[61]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


# In[62]:


clf_all_for = RandomForestClassifier(random_state=0).fit(X_train_all, y_train_all)
acc_all_for = clf_all_for.score(X_test_all, y_test_all)
print ('Random Forest Accuracy (Default parameters): {}'.format(acc_all_for))


# In[63]:


parameters_f = [{'max_depth': range(2,10), 'n_estimators': range(2,8,2), 'max_features': range(10,20)}]
clf_all_for_g = GridSearchCV(RandomForestClassifier(random_state=0), parameters_f)
clf_all_for_g.fit(X_train_all, y_train_all)

print('Best score for train data:', clf_all_for_g.best_score_)
print('Best depth:',clf_all_for_g.best_estimator_.max_depth)
print('Best n trees:',clf_all_for_g.best_estimator_.n_estimators)
print('Best n features:',clf_all_for_g.best_estimator_.max_features)
print('Score for test data:',clf_all_for_g.score(X_test_all, y_test_all))


# In[64]:


clf_all_nn = MLPClassifier(random_state=0).fit(X_train_all, y_train_all)
acc_all_nn = clf_all_nn.score(X_test_all, y_test_all)
print ('Neural Networks Accuracy (Default parameters): {}'.format(acc_all_nn))

parameters_n = [{'alpha': [0.0001, 0.001, 0.01, 0.1], 'hidden_layer_sizes':[(10,),(20,),(100,)]}]
clf_all_nn_g = GridSearchCV(MLPClassifier(random_state=0), parameters_n)
clf_all_nn_g.fit(X_train_all, y_train_all)

print('Best score for train data:', clf_all_nn_g.best_score_)
print('Best alpha:',clf_all_nn_g.best_estimator_.alpha)
print('Best hidden_layer_sizes:',clf_all_nn_g.best_estimator_.hidden_layer_sizes)
print('Score for test data:',clf_all_nn_g.score(X_test_all, y_test_all))


# In[67]:


y3 = ml3['Position']
X3 = ml3[['Crossing', 
           'Finishing',
           'HeadingAccuracy',
           'ShortPassing', 
           'Volleys', 
           'Dribbling',
           'Curve',
           'FKAccuracy',
           'LongPassing',
           'BallControl',
           'Acceleration',
            'SprintSpeed',
           'Agility',
           'Reactions', 
           'Balance',
           'ShotPower', 
           'Jumping',
           'Stamina', 
           'Strength',
           'LongShots',
           'Aggression',
           'Interceptions',
           'Positioning', 
           'Vision', 
           'Penalties',
           'Composure',
           'Marking',
           'StandingTackle', 
           'SlidingTackle',]]
X_tr, X_tt, y_tr, y_tt = train_test_split( X3,y3, random_state=24)


# In[68]:


print('X train shape: {}'.format(X_tr.shape))
print('X test shape: {}'.format(X_tt.shape))
print('y train shape: {}'.format(y_tr.shape))
print('y test shape: {}'.format(y_tt.shape))


# In[71]:



clf_d1 = DummyClassifier(strategy = 'most_frequent').fit(X_tr, y_tr)
acc_d1 = clf_d1.score(X_tt, y_tt)
print ('Dummy Classifier (most frequent class): {}'.format(acc_d1))



# In[72]:


clf1 = LogisticRegression().fit(X_tr, y_tr)
acc1 = clf1.score(X_tt, y_tt)
print ('Logistic Regression Accuracy: {}'.format(acc1))


# In[74]:


clf_all_for1 = RandomForestClassifier(random_state=0).fit(X_tr, y_tr)
acc_all_for1 = clf_all_for1.score(X_tt, y_tt)
print ('Random Forest Accuracy (Default parameters): {}'.format(acc_all_for1))


# In[75]:


parameters_f1 = [{'max_depth': range(2,10), 'n_estimators': range(2,8,2), 'max_features': range(10,20)}]
clf_all_for_g1 = GridSearchCV(RandomForestClassifier(random_state=0), parameters_f1)
clf_all_for_g1.fit(X_tr, y_tr)

print('Best score for train data:', clf_all_for_g1.best_score_)
print('Best depth:',clf_all_for_g1.best_estimator_.max_depth)
print('Best n trees:',clf_all_for_g1.best_estimator_.n_estimators)
print('Best n features:',clf_all_for_g1.best_estimator_.max_features)
print('Score for test data:',clf_all_for_g1.score(X_tt, y_tt))


# In[77]:


clf_all_nn1 = MLPClassifier(random_state=0).fit(X_tr, y_tr)
acc_all_nn1 = clf_all_nn1.score(X_tt, y_tt)
print ('Neural Networks Accuracy (Default parameters): {}'.format(acc_all_nn1))

parameters_n1 = [{'alpha': [0.0001, 0.001, 0.01, 0.1], 'hidden_layer_sizes':[(10,),(20,),(100,)]}]
clf_all_nn_g1 = GridSearchCV(MLPClassifier(random_state=0), parameters_n1)
clf_all_nn_g1.fit(X_tr, y_tr)

print('Best score for train data:', clf_all_nn_g1.best_score_)
print('Best alpha:',clf_all_nn_g1.best_estimator_.alpha)
print('Best hidden_layer_sizes:',clf_all_nn_g1.best_estimator_.hidden_layer_sizes)
print('Score for test data:',clf_all_nn_g1.score(X_tt, y_tt))


# In[ ]:





# In[ ]:




