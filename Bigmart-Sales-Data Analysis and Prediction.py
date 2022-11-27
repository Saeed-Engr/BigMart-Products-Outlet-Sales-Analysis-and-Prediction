#!/usr/bin/env python
# coding: utf-8

# Auther Name:        Muhammad Saeed
# -
# - Github profile:     https://github.com/Saeed-Engr
# - Linkedin Profile:   https://www.linkedin.com/in/saeedengr

# Project Description
# -
# The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim of this data science project is to build a predictive model and find out the sales of each product at a particular store.
# 
# Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.
# 
#  The data has missing values as some stores do not report all the data due to technical glitches. Hence, it will be required to treat them accordingly.

# In[ ]:





# # 1).Problem Statement

# Bigmart have collected 2013 sales data for 1559 products across 10 stores in different cities. The aim of this data science project is to build a predictive model and find out the sales of each product at a particular store.

# # 2).Hypothesis Generation

# ## **2. Hypothesis generation with respect to problem statement.**
# 
# 1. Item weight might effect a sales of the product.
# 2. Sales of the product may be depends on the items fat content.
# 3. More Item_Visibility of a particular product may be costlier than other products.
# 4. Item type could have an effect on the sales.
# 5. Are the items with more MRP have more item outlet sales.
# 6. Are the stores which have established earlier have more sales.
# 7. Size of the stores could have an effect on the item sales at a particular store.
# 8. Location of the stores might depends on the Item outlet sales.
# 9. Are the supermarkets have more sales than others.
# 

# # 3).Loading Packages and Data

# In[167]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os #paths to file
import numpy as np # linear algebra
import pandas as pd # data processing
import warnings# warning filter


#ploting libraries
import matplotlib.pyplot as plt 
import seaborn as sns

#feature engineering
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#train test split
from sklearn.model_selection import train_test_split

#metrics
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.model_selection  import cross_val_score as CVS


#ML models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso


#default theme and settings
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)
pd.options.display.max_columns

#warning hadle
warnings.filterwarnings("always")
warnings.filterwarnings("ignore")


# In[168]:


train = pd.read_csv("E:/Technocolabs Internship/1st mini project/Train.csv")
test = pd.read_csv("E:/Technocolabs Internship/1st mini project/Test.csv")
train.head()


# In[169]:


test.head()


# In[170]:


print(train.shape)
print(test.shape)


# ### **Observations:**
# * This shows that the train and test data is imported successfully.
# * The train data consists of 8,523 training examples with 12 features.
# * The test data consists of 5,681 training examples with 11 features
# 

# In[ ]:





# # 4).Data Structure and Content

# In[171]:


print(train.ndim)
print(test.ndim)


# In[172]:


print(train.columns)


# In[173]:


print(test.columns)


# In[174]:


train.info()


# In[175]:


cat_features = [feature for feature in train.columns if train[feature].dtypes == 'O']
print('Number of categorical variables: ', len(cat_features))
print('*'*80)
print('Categorical variables column name:',cat_features)


# In[176]:


numerical_features = [feature for feature in train.columns if train[feature].dtypes != 'O']
print('Number of numerical variables: ', len(numerical_features))
print('*'*80)
print('Numerical Variables Column: ',numerical_features)


# Numerical Features:
# -
# * Item_Weight
# * Item_Visibility
# * Item_MRP
# * Outlet_Establishment_Year
# * Item_Outlet_Sales(Target Variable)
# 
# ### **Categorical Features**
# * Item_Identifier
# * Item_Fat_Content(Ordinal Feature)
# * Item_Type
# * Outlet_Itemtifier
# * Outlet_Size(Ordinal Feature)
# * Outlet__Location_Type(Ordinal Feature)
# * Ootlet_Type(Ordinal Feature)
# 
# **Observations:**
# * There are 4 float type variables, 1 integer type and 7 object type.
# * We are considering Item_Establishment_Year as a categorical feature because it contains some fixed value but not converting its data type now will consider later.
# * Item_Fat_Content, Outlet_Size, Outelet_Location_Type and Outlet_Type are ordinal features because these values can be arranged in some order.
# 
# 

# In[ ]:





# # 5).Exploratory Data Analysis

# In[177]:


train.describe()


# In[178]:


train.isnull().any()


# In[179]:


val = train.isnull().sum()
val


# In[180]:


total_cells = np.product(train.shape)
total_cells


# In[181]:


train.shape


# In[182]:


8523*12


# In[183]:


#percent of data that is missing

# how many total missing values do we have?
total_cells = np.product(train.shape)
total_missing = val.sum()

# percent of data that is missing
percent_missing = (total_missing/total_cells) * 100
print(percent_missing)


# In[184]:


train.columns


# In[185]:


train.dtypes


# In[186]:


train.info()


# In[187]:


train.tail()


# In[188]:


train.columns


# In[189]:


train.duplicated().sum()


# In[190]:


train['Item_Outlet_Sales'].nunique()


# In[191]:


train.sample(5)


# In[192]:


train["Item_Fat_Content"].nunique()


# In[193]:


train["Item_Fat_Content"].unique()


# In[194]:


train["Item_Type"].nunique()


# In[195]:


train["Item_Type"].unique()


# In[196]:


def bar_plot(variable):
    
    var=train[variable]
    var_Value=var.value_counts()
    
    #visualize
    plt.figure(figsize=(9,3))
    plt.bar(var_Value.index,var_Value.values)
    
    plt.xlabel("Item_Outlet_Sales")
    plt.ylabel("Item_Weight")
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable,var_Value))

###########################################    

for val in numerical_features:
    bar_plot(val)


# In[197]:


#category2=["Gender", "Customer Type", "Type of Travel", "Class","Satisfaction"]

for c in cat_features:
    print("{} \n".format(train[c].value_counts()))


# In[198]:


train.Item_Weight.describe()


# Firstly we need to split our data to categorical and numerical data,
# 
# using the .select_dtypes('dtype').columns.to_list() combination.

# In[199]:


#list of all the numeric columns
num = train.select_dtypes('number').columns.to_list()
#list of all the categoric columns
cat = train.select_dtypes('object').columns.to_list()

#numeric df
BM_num =  train[num]
#categoric df
BM_cat = train[cat]

#print(num)
#print(cat)

[train[category].value_counts() for category in cat[1:]]


# In[200]:


#train
train['Item_Fat_Content'].replace(['LF', 'low fat', 'reg'], 
                                  ['Low Fat','Low Fat','Regular'],inplace = True)

#check result
train.Item_Fat_Content.value_counts()


# In[201]:


train.head()


# `Outlet_Establishment_Year` is quite useless as it is, making a new column with the age the new name will be `Outlet_Age`

# In[202]:


#creating our new column for train datasets
train['Outlet_Age']= train['Outlet_Establishment_Year'].apply(lambda year: 2020 - year)
##uncomment to check result
#tr_df['Outlet_Age'].head
#te_df['Outlet_Age'].head


# In[203]:


train.head(2)


# # Data Visualization

# Univariate Plots
# -
# For starters we will create countplots for the categorical columns:

# In[204]:


#categorical columns:
['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 
 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
 
plt.figure(figsize=(6,4))
sns.countplot(x='Item_Fat_Content' , data=train ,palette='mako')
plt.xlabel('Item_Fat_Content', fontsize=14)
plt.show()


# In[205]:


plt.figure(figsize=(20,10))
sns.countplot(x='Item_Type' , data=train ,palette='summer')
plt.xlabel('Item_Type', fontsize=14)
plt.show()


# In[206]:


plt.figure(figsize=(15,4))
sns.countplot(x='Outlet_Identifier' , data=train ,palette='winter')
plt.xlabel('Outlet_Identifier', fontsize=14)
plt.show()


# In[207]:


plt.figure(figsize=(10,4))
sns.countplot(x='Outlet_Size' , data=train ,palette='autumn')
plt.xlabel('Outlet_Size', fontsize=14)
plt.show()


# In[208]:


plt.figure(figsize=(10,4))
sns.countplot(x='Outlet_Location_Type' , data=train ,palette='twilight_shifted')
plt.xlabel('Outlet_Location_Type', fontsize=14)
plt.show()


# In[209]:


plt.figure(figsize=(10,4))
sns.countplot(x='Outlet_Type' , data=train ,palette='rocket')
plt.xlabel('Outlet_Type', fontsize=14)
plt.show()


# Categoric columns realizations
# -
# * `Item_Fat_Content` - Most items sold are low fat.
# * `Item_Type` - Item types that are distictly popular are `fruits and vegetables` and `snack foods`. 
# * `Outlet_Identifier` - Sold items are ditributed evenly among outlets excluding `OUT010` and `OUT019` that are significanly lower. 
# * `Outlet_Size` - Bigmart outlets are mostly medium sized in our data. 
# * `Outlet_Location_Type` - The most common type is `Tier3`.
# * `Outlet_Type` - By a wide margin the mode outlet type is `Supermarket Type1`.
# 
# Now for the numerical columns:
# -
# 

# In[210]:


#list of all the numeric columns
num = train.select_dtypes('number').columns.to_list()
#numeric df
BM_num =  train[num]

plt.hist(train['Outlet_Age'])
plt.title("Outlet_Age")
plt.show()


# In[211]:


#because of the variability of the unique values of the numeric columns a scatter plot with the target value will be of use
for numeric in BM_num[num[:3]]:
    plt.scatter(BM_num[numeric], BM_num['Item_Outlet_Sales'])
    plt.title(numeric)
    plt.ylabel('Item_Outlet_Sales')
    plt.show()


#  multivariate plots
# -
# I want to check the following relationships with `Item_Outlet_Sales`:
# * Sales per item type
# * Sales per outlet
# * Sales per outlet type
# * Sales per outlet size
# * Sales per location type

# In[212]:


plt.figure(figsize=(27,10))
sns.barplot('Item_Type' ,'Item_Outlet_Sales', data=train ,palette='gist_rainbow_r')
plt.xlabel('Item_Type', fontsize=14)
plt.legend()
plt.show()


# In[213]:


plt.figure(figsize=(27,10))
sns.barplot('Outlet_Identifier' ,'Item_Outlet_Sales', data=train ,palette='gist_rainbow')
plt.xlabel('Outlet_Identifier', fontsize=14)
plt.legend()
plt.show()


# In[214]:


plt.figure(figsize=(10,5))
sns.barplot('Outlet_Type' ,'Item_Outlet_Sales', data=train ,palette='nipy_spectral')
plt.xlabel('Outlet_Type', fontsize=14)
plt.legend()
plt.show()


# In[215]:


plt.figure(figsize=(10,5))
sns.barplot('Outlet_Size' ,'Item_Outlet_Sales', data=train ,palette='YlOrRd')
plt.xlabel('Outlet_Size', fontsize=14)
plt.legend()
plt.show()


# In[216]:


plt.figure(figsize=(10,5))
sns.barplot('Outlet_Location_Type' ,'Item_Outlet_Sales', data=train ,palette='Spectral')
plt.xlabel('Outlet_Location_Type', fontsize=14)
plt.legend()
plt.show()


# Realizations:
# -
# * The difference in item types by sales is very small.
# * Outlet 27 is the most profitable and there is a big diffrence between each specific outlet sales.
# * Suprisingly supermarket type 3 is the most profitable and not type 1.
# * Medium and high outlet sizes are pretty much even in sales.
# * Tier 2 and 3 are almost even being the highest in sales (2 is slightly larger). 

# # correlation matrix

# In[217]:


#plotting the correlation matrix
sns.heatmap(train.corr() ,cmap='rocket')


# # Missing Values

# In[ ]:





# In[218]:


plt.figure(figsize = (10,6))
sns.heatmap(train.isnull(), yticklabels=False,cbar = False,cmap ='viridis')


# ### **Observation:**
# * Yes our dataset contain missing values.
# * Item_weight and Outlet_Size features contain missing values

# * **Let's calculate the percentage of missing values**

# In[219]:


def missing_percent():
  miss_item_weight = (train['Item_Weight'].isnull().sum()/len(train))*100
  miss_Outlet_Size = (train['Outlet_Size'].isnull().sum()/len(train))*100

  print('% of missing values in Item_Weight: ' + str(miss_item_weight))
  print('% of missing values in Outlet_Size: ' +str(miss_Outlet_Size))


# In[220]:


missing_percent()


# ### **Observations:**
# * Since the percentage of missing values is very high so we can't drop these values otherwise we can miss some important information. Only way is to handle the missing values using some technique.
# 
# ### **Things to invetigate:**
# * Do the missing values of Item weight have some relation with sales of the items or any other feature.
# * Do the missing values of Outlet size have some relation with any other feature.
# 

# In[221]:


sns.displot(
    data=train.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=2
)


# Fill Nan Values with median
# -
# 

# In[222]:


train["Item_Weight"]=train["Item_Weight"].fillna(train["Item_Weight"].mean())


# In[223]:


sns.displot(
    data=train.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=2
)


# Outlet_Size
# -
# `Outlet_Size` is a categorical column, therefore we will impute missing values with `Medium` the **mode value**

# In[224]:


print("train mode", [train['Outlet_Size'].mode().values[0]])
       


# In[225]:


#train
train['Outlet_Size'] = train['Outlet_Size'].fillna(
train['Outlet_Size'].dropna().mode().values[0])

#checking if we filled missing values
train['Outlet_Size'].isnull().sum()


# In[226]:


sns.displot(
    data=train.isna().melt(value_name="missing"),
    y="variable",
    hue="missing",
    multiple="fill",
    aspect=2
)


# In[227]:


# I personally prefer a vertical view and a cyan color
sns.boxplot(data=train['Item_Weight'],orient="v", color = 'c')
plt.title("Item_Weight Boxplot")


# In[ ]:





# In[228]:


train.isnull().sum()


# # Feature Engineering
# 

# ### Feature Engineering
# 
# **Categorical values**:
# 
# We have 7 columns we need to delete or encode.
# 
# * Ordinal variables:
#     * `Item_Fat_Content`  
#     * `Outlet_Size`  
#     * `Outlet_Location_Type`
#     
# * Nominal variables:
#     * `Item_Identifier `  
#     * `Item_Type`
#     * `Outlet_Identifier`
#     * `Outlet_Type`
# 
# **Numeric values**:
# 
# * From the numeric variables `Outlet_Establishment_Year` is no longer needed
# 
# **Conclusion:**
# 
# In my FE process i have decided:
# 
# 1. The columns `Outlet_Establishment_Year`, `Item_Identifier ` and `Outlet_Identifier` don't have significant values so we will drop them.
# 2. All Ordinal variables will be Label encoded.
# 3. The columns `Outlet_Type` and `Item_Type`  will be One Hot encoded.
# 
# 

# In[251]:


# Dropping irrelevant columns

train  = train.drop(['Item_Identifier'],axis=1)

#te_fe = te_fe.drop(['Item_Identifier','Outlet_Identifier','Outlet_Establishment_Year','Outlet_Type','Item_Type'],axis=1)


# In[255]:


train_data_num = train.select_dtypes("float64")
train_data_cat = train.select_dtypes("object")


# In[256]:


train_data_num.head(2)


# In[257]:


train_data_cat.head(2)


# In[258]:


train_data_cat.shape


# # Categorical encoding

# Dropping Irrelevant Columns
# -
# 

# In[260]:


train_data_cata_encoded=pd.get_dummies(train_data_cat, columns=train_data_cat.columns.to_list())
train_data_cata_encoded.head()


# # Concatenate Data

# In[261]:


data=pd.concat([train_data_cata_encoded,train_data_num],axis=1,join="outer")
data.head()


# In[262]:


data.columns


# # Buid Models

# In[264]:


data.shape


# In[268]:


y = data['Item_Outlet_Sales']
x = data.drop('Item_Outlet_Sales', axis = 1)


# In[269]:


print(x.shape)
print(y.shape)


# In[270]:


x.head(2)


# In[272]:


y.head(5)


# In[273]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[274]:


def cross_val(model_name,model,X,y,cv):
    
    scores = CVS(model, X, y, cv=cv)
    print(f'{model_name} Scores:')
    for i in scores:
        print(round(i,2))
    print(f'Average {model_name} score: {round(scores.mean(),4)}')


# # Linear Regression

# In[281]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics


# ## Linear Regression
# 
# ![](https://cdn.filestackcontent.com/WCbMsxiSLW2H1SyqunQm)
# 
# In statistics, linear regression is a linear approach to modelling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables).
# 
# Linear regression was the first type of regression analysis to be studied rigorously, and to be used extensively in practical applications. This is because models which depend linearly on their unknown parameters are easier to fit than models which are non-linearly related to their parameters and because the statistical properties of the resulting estimators are easier to determine.
# 

# In[282]:


#model
LR = LinearRegression(normalize=True)

#fit
LR.fit(X_train, y_train)

#predict
y_predict = LR.predict(X_test)

#score variables
LR_MAE = round(MAE(y_test, y_predict),2)
LR_MSE = round(MSE(y_test, y_predict),2)
LR_R_2 = round(R2(y_test, y_predict),4)
LR_CS  = round(CVS(LR, X, y, cv=5).mean(),4)

print(f" Mean Absolute Error: {LR_MAE}\n")
print(f" Mean Squared Error: {LR_MSE}\n")
print(f" R^2 Score: {LR_R_2}\n")
cross_val(LR,LinearRegression(),X,y,5)


# In[283]:


Linear_Regression=pd.DataFrame({'y_test':y_test,'prediction':y_predict})
Linear_Regression.head()


# In[284]:


def evaluate(y_test,y_predict):
    print("MAE ",metrics.mean_absolute_error(y_predict,y_test))
    print("MSE",metrics.mean_squared_error(y_predict,y_test))
    print("RMSE ",np.sqrt(metrics.mean_squared_error(y_predict,y_test)))


# In[288]:


lr = LinearRegression().fit(X_train, y_train)
accuracy = lr.score(X_train, y_train)
print(accuracy)

###########################

y_predict=lr.predict(X_test)
evaluate(y_test,y_predict)


# ## Random Forest Regressor
# 
# ![](https://lh3.googleusercontent.com/proxy/V_3AWj1s3kBvrcJEUczXaoNlIVmToUBGxo_wuNSM2B3NNUs1q31KuEETmfxw3jIfiJ5H3SkjTCs9rq8BOgRZnP-ZIZBjLwVMRMchhNeV0SJQknEdTd4dhjrULXqViViMORUPWvoGMQuGYol-sj5lIEXKuHo4ouNxp3-m-sOUkzDbZ10Ph-a769ugPqsTLvJLlfhr0sbSz6Y0=s0-d)
# 
# Random forest is a Supervised Learning algorithm which uses ensemble learning method for classification and regression.
# It operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

# In[291]:


#model
RFR= RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4,random_state=101)
#fit
RFR.fit(X_train, y_train)
#predict
y_predict = RFR.predict(X_test)

#score variables
RFR_MAE = round(MAE(y_test, y_predict),2)
RFR_MSE = round(MSE(y_test, y_predict),2)
RFR_R_2 = round(R2(y_test, y_predict),4)
RFR_CS  = round(CVS(RFR, X, y, cv=5).mean(),4)



print(f" Mean Absolute Error: {RFR_MAE}\n")
print(f" Mean Squared Error: {RFR_MSE}\n")
print(f" R^2 Score: {RFR_R_2}\n")
cross_val(RFR,RandomForestRegressor(),X,y,5)


# In[292]:


Random_Forest_Regressor=pd.DataFrame({'y_test':y_test,'prediction':y_predict})
Random_Forest_Regressor.head()


# In[293]:


from sklearn.ensemble import RandomForestRegressor

random_cl = RandomForestRegressor(n_estimators=500)
random_cl=random_cl.fit(X_train,y_train)
print(random_cl)

#######################
y_predict=random_cl.predict( X_test)
evaluate(y_predict,y_test)


# In[ ]:





# # xgboost

# In[294]:


import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

xgb_r = xgb.XGBRegressor(objective ='reg:linear',n_estimators = 100, seed = 123)

# Fitting the model
xgb_r=xgb_r.fit(X_train, y_train)
 


# In[295]:


# Predict the model
y_predict = xgb_r.predict(X_test)
evaluate(y_test,y_predict)


# In[ ]:





# ## Lasso Regressor
# 
# ![](data:image/png;base64,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)
# 
# In statistics and machine learning, lasso (least absolute shrinkage and selection operator; also Lasso or LASSO) is a regression analysis method that performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the statistical model it produces  - more info on [wikipedia.](https://en.wikipedia.org/wiki/Lasso_(statistics)) 
# 

# In[296]:


#model
LS = Lasso(alpha = 0.05)
#fit
LS.fit(X_train,y_train)

#predict
y_predict = LS.predict(X_test)

#score variables
LS_MAE = round(MAE(y_test, y_predict),2)
LS_MSE = round(MSE(y_test, y_predict),2)
LS_R_2 = round(R2(y_test, y_predict),4)
LS_CS  = round(CVS(LS, X, y, cv=5).mean(),4)

print(f" Mean Absolute Error: {LS_MAE}\n")
print(f" Mean Squared Error: {LS_MSE}\n")
print(f" R^2 Score: {LS_R_2}\n")
cross_val(LS,Lasso(alpha = 0.05),X,y,5)


# # Conclusion

# In[297]:


MAE= [LR_MAE,RFR_MAE,LS_MAE]
MSE= [LR_MSE,RFR_MSE,LS_MSE]
R_2= [LR_R_2,RFR_R_2,LS_R_2]
Cross_score= [LR_CS,RFR_CS,LS_CS]

Models = pd.DataFrame({
    'models': ["Linear Regression","Random Forest Regressor","Lasso Regressor"],
    'MAE': MAE, 'MSE': MSE, 'R^2':R_2, 'Cross Validation Score':Cross_score})
Models.sort_values(by='MAE', ascending=True)


# ## Summary
# 
# * `Item_MRP`  optimizes Maximum Outlet sales (positive correlation with the target).
# * Linear Regression	and Lasso Regressor have the best perfomance in most categories.
# * only a third of the observed variation can be explained by the model's inputs of Random Forest Regressor, there for it's performance is not optimal even though his cross validation is the highest.
# * For better peformance this models need tuning e.g. Grid Search.

# In[ ]:




