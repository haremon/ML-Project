#!/usr/bin/env python
# coding: utf-8

# # FDS PROJECT

# ###  We will be predicting the costs of used cars in the given dataset.

# **Steps:** <br>
# 
# 1.Clean Data(Null value removal, Outlier identification)<br>
# 
# 2.Null Values(Dropping the rows /Columns).<br>
# 
# 3.EDA: To understand the relations between columns.<br>
# 
# 4.Labing Encoding for Categorical Data.<br>
# 
# 5.The train test split.<br>
# 
# 7.Applying different ML regression Algorithms.<br>
# 
# 8.Calculating the error metrics.<br>

# **Importing libraries**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# **Importing the Dataset**

# In[2]:


path="C:/Users/Shivam/Downloads/Data_Train.xlsx"


# In[3]:


data=pd.read_excel(path)


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.dtypes


# In[8]:


data.shape


# In[9]:


data.columns 


# In[10]:


data.isnull().sum()


# 
# This shows us the number of null values in each column.<br>
# We see that columns- Mileage(2),Engine(36),Power(36),Seats(42) have null values.<br>
# Further, we will see whether we need to drop columns or replace the null values with some other reasonable value

# In[11]:


import numpy as np

# replace "?" to NaN
data.replace("?", np.nan, inplace = True)


# We replace "?" with NaN (Not a Number), Python's default missing value marker for reasons of computational speed and convenience.

# **Removing Duplicate values**

# In[12]:


duplicate=data.drop_duplicates(['Name','Location','Year','Kilometers_Driven','Fuel_Type','Transmission','Owner_Type','Mileage','Engine','Power','Seats'])


# 

# Lets drop the km/l and km/kg from the mileage column and convert it to float so that the further calculations become easy.

# In[13]:


data['Mileage'] = data['Mileage'].apply(lambda x : str(x).split(' ')[0]).astype(float)


# In[14]:


data.head()


# Changing engine and power datatype to float from object by dropping CC and bhp respectively.

# In[15]:


data['Engine'] = data['Engine'].apply(lambda x : str(x).split(" ")[0]).astype(float)
data['Power'] = data['Power'].replace('null bhp','0 bhp').apply(lambda x : str(x).split(' ')[0]).astype(float)
data.head()


# **Handling null values**

# In[16]:


data.isnull().sum()


# As we can see that the mileage column has only two null values,so we remove the rows here.

# In[17]:


data = data[data.Mileage.notnull()]
data.isnull().sum()


# In[18]:


missing_data = data.isnull()
missing_data.head(20)


# In[19]:


for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")   


# In[ ]:





# Checking if the engine column has outlier values or not using boxplot

# In[20]:


sns.boxplot('Engine',data=data)
Q1 = data.Engine.quantile(0.25)
print(Q1)
Q3 = data.Engine.quantile(0.75)
print(Q3)
IQR = Q3 - Q1


# So the above boxplot clearly shows that there are approx 12 outlier values present in the column 'Engine',so we replace the null values with the median of the column.

# In[21]:


print(data["Engine"].dtype)
avg_norm_loss = data["Engine"].astype("float").median(axis=0)

print("Median of Engine:", avg_norm_loss)


# In[22]:


data['Engine'] = data['Engine'].fillna(data.Engine.median())


# In[23]:


data.isnull().sum()


# Now, we will check the outliers for power column and accordingly will replace the null values with mean or median of the column.

# In[24]:


sns.boxplot('Power',data=data)
Q1 = data.Power.quantile(0.25)
print(Q1)
Q3 = data.Power.quantile(0.75)
print(Q3)
IQR = Q3 - Q1


# As we can see that the power columns has lot of outlier values so we will replace the null null values with the median of the column.

# In[25]:


print(data["Power"].dtype)
avg_norm_loss = data["Power"].astype("float").median(axis=0)

print("Median of Power:", avg_norm_loss)


# In[66]:


data['Power'] = data['Power'].fillna(data.Engine.median())


# In[67]:


data.isnull().sum()


# Now we will repeat the same process for the seats column.

# In[28]:


sns.boxplot('Seats',data=data)
Q1 = data.Seats.quantile(0.25)
print(Q1)
Q3 = data.Seats.quantile(0.75)
print(Q3)
IQR = Q3 - Q1


# As we can see clearly from the above boxplot that Seats column also has oultier value,so we replace the null values with the median of the seats column.

# In[29]:


print(data["Seats"].dtype)
avg_norm_loss = data["Seats"].astype("float").median(axis=0)

print("Average of Seats:", avg_norm_loss)


# In[30]:


data['Seats'] = data['Seats'].fillna(data.Seats.median())


# In[31]:


data.isnull().sum()


# So,now we have removed all the null values from our dataframe,so our data is now ready to be processed for further analysis.

# In[32]:


data.isnull().values.any()


# In[33]:


data.isnull()


# We have now treated the null values, by replacing them with the reasonable value

# ## EXPLORATORY DATA ANALYSIS

# In[34]:


#Uderstanding the basic information of the data like min, max, mean and standard deviation etc.
data.corr()


# In[35]:


data.min()


# In[36]:


data.max()


# In[37]:


#This displays general information about the dataset with informations like the column names their data types 
#and the count of non-null values for every column.
data.info()


# In[38]:


#Gives the data types of all the columns values in the dataframe
data.dtypes


# As we can see the columns: Name, Location, Fuel_type, Transmission and Owner Type are CATERGORICAL Variables.<br>
# As we can see the columns: Year, Kilometers_Driven, Mileage, Engine, Power, Seats and Price are NUMERICAL Variables.

# #### Analysing the values of all CATEGORICAL Variables and check the data type of values

# In[39]:


#Gives the types of categories present in each categorical variable 
data['Name'].value_counts()


# In[40]:


data['Name'].value_counts().to_frame()


# In[41]:


#Gives the types of categories present in each categorical variable 
data['Location'].value_counts()


# In[42]:


data['Location'].value_counts().to_frame()


# In[43]:


#Gives the types of categories present in each categorical variable 
data['Fuel_Type'].value_counts()


# In[44]:


#Gives the types of categories present in each categorical variable 
data['Transmission'].value_counts()


# In[45]:


#Gives the types of categories present in each categorical variable 
data['Owner_Type'].value_counts()


# ## DATA VISUALIZATION

# In[46]:


# Univariate visualisation for quantative features
features = ['Kilometers_Driven', 'Mileage','Power','Seats']
data[features].hist(figsize=(15, 8));


# In[47]:


sns.regplot(x="Seats", y="Price", data=data)


# Weak Linear Relationship

# In[48]:


sns.regplot(x="Year", y="Price", data=data)


# In[49]:


p = sns.countplot(x="Price", data = data, palette="muted")
_ = plt.setp(p.get_xticklabels(), rotation=90) 


# In[50]:


sns.pairplot(data=data,plot_kws={'alpha':0.2})


# In[51]:


columns = ['Name', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission','Owner_Type','Mileage','Engine','Power','Seats']
for i, col in enumerate(columns):
    counts = data.groupby(by=col)[col].count().sort_values(ascending=False)
    cat = counts.index
    r = range(len(cat))
    plt.figure()
    plt.title(col)
    plt.bar(r, counts)
    plt.xticks(r, cat)
    plt.show()
    print(cat)


# In[52]:


#multivariate visualisation 
sns.boxplot(x = 'Fuel_Type', y = 'Power', data = data) 


# In[53]:


sns.boxplot(x = 'Transmission', y = 'Mileage', data = data)


# In[54]:


sns.boxplot(x = 'Seats', y = 'Price', data= data)


# In[55]:


sns.boxplot(x = 'Fuel_Type', y = 'Price', data= data)


# In[56]:


plt.subplot(1,2,1)
plt.title('Fuel Type Histogram')
sns.countplot(data.Fuel_Type, palette=("Blues_d"))


# In[57]:


sem_price = data.groupby(['Location'])['Price'].sem().reset_index()
plt.figure(figsize=(20, 10));
sns.set_style("ticks", {"xtick.major.size": 16, "ytick.major.size":8});
sns.set(font_scale=1.1)
fig = sns.barplot(x=data['Location'],y= data['Price'],yerr=sem_price['Price'],errwidth=3,palette="Blues_d")
plt.ylabel('Price: Rupees',fontsize=16);
plt.xlabel('Location',fontsize=16);
plt.xticks()
plt.title('Price per Location',fontsize=22,fontweight='bold');


# In[58]:


mx = sns.factorplot(x="Year", y="Price", data=data, kind="box", size=10 ,aspect=3)
mx.set(xlabel= 'Year')
mx.set_xticklabels('Year',rotation=45)
mx.set(ylabel= 'Price')
plt.title('Year Vs Price');
plt.show()


# In[ ]:


from pandas.plotting import scatter_matrix
attributes = ["Price", "Seats", "Year"]
scatter_matrix(data[attributes], figsize=(12,18), diagonal='scatter')


# In[ ]:


plt.figure(figsize =(8,8))
plt.title('Feature Correlation Heat Map')
sns.heatmap(data.corr(),linewidths=.1,vmax=1.0,
            square=True,linecolor='',annot=True)


# From the correlation heat map above, it can be observed that power of the car and Engine have a major influence on the price.

# In[59]:


from scipy import stats


# In[60]:


pearson_coef, p_value = stats.pearsonr(data['Engine'], data['Price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# In[61]:


pearson_coef, p_value = stats.pearsonr(data['Seats'], data['Price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# In[68]:


pearson_coef, p_value = stats.pearsonr(data['Power'], data['Price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# In[69]:


#skewness and kurtosis
print("Skewness: %f" % data['Price'].skew())
print("Kurtosis: %f" % data['Price'].kurt())


# In[70]:


df2=data[['Engine','Power','Seats','Year']]
sns.pairplot(df2, hue=None,size=3)


# In[71]:


sns.distplot(data['Price'])


# In[72]:


plt.figure(figsize=(14,6))
plt.subplot(1, 4, 1)
plt.title('Price')
sns.violinplot(y='Price',data=data,palette='summer',linewidth=3)
plt.show()


# ## LABEL ENCODING

# In[73]:


data.dtypes


# In[74]:


data["Name"] = data["Name"].astype('category')
data.dtypes


# In[75]:


data["Name_Cat"] = data["Name"].cat.codes
data.head()


# In[76]:


data["Location"] = data["Location"].astype('category')
data["Location_Cat"] = data["Location"].cat.codes
data.head()


# In[77]:


data["Fuel_Type"] = data["Fuel_Type"].astype('category')
data["Fuel_Type_Cat"] = data["Fuel_Type"].cat.codes
data.head()


# In[78]:


data["Transmission"] = data["Transmission"].astype('category')
data["Transmission_Cat"] = data["Transmission"].cat.codes
data.head()


# In[79]:


data["Owner_Type"] = data["Owner_Type"].astype('category')
data["Owner_Type_Cat"] = data["Owner_Type"].cat.codes
data.head()


# In[80]:


data.dtypes


# ## DATA SCALING

# In[81]:


#using normalization formula:X-X(min)/X(max)-X(min)
M=data['Kilometers_Driven'].max() 
m=data['Kilometers_Driven'].min()
data['Kilometers_Driven']=(data['Kilometers_Driven']-m)/(M-m)


# In[82]:


data.head(10)


# We have seen that by scaling the column Kilometer_Driven the values of that column are ranging from 0 to 1, which balances all values.

# In[83]:


data=data.drop(['Name','Location','Fuel_Type','Transmission','Owner_Type'],axis=1)


# In[84]:


data.head(10)


# ## TRAIN TEST SPLIT

# In[85]:


X= data.drop(labels = ["Price"],axis = 1)
y=data.Price


# In[86]:


from sklearn.model_selection import train_test_split
data.loc[:,data.columns!='Price'],data['Price'],
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)


# In[87]:


data.isnull().any().any()


# In[88]:


# X = data.drop('Price',axis=1)
# y = data['Price']


# In[89]:


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 50)


# In[90]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn import tree
from sklearn.model_selection import train_test_split
from six import StringIO 
from IPython.display import Image  
from sklearn.tree import export_graphviz
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.metrics import mean_squared_error


# ## ML REGRESSION ALGORITHMS

# ### 1. LINEAR REGRESSION:

# In[118]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
y_pred= linear_reg.predict(X_test)
print("The score on the training set with linear regression is: ",linear_reg.score(X_train,y_train))
print("The score on the testing set with linear regression is: ",linear_reg.score(X_test,y_test))
print("Root Mean squared error: %.2f"
      % np.sqrt(mean_squared_error(y_test, y_pred)))
print("Mean squared error:"
      , mean_squared_error(y_test, y_pred))


# In[125]:


from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error
print("R2 score:",r2_score(y_pred, y_test))
y_pred_dt=linear_reg.predict(X_test)
from sklearn.metrics import mean_squared_error
print("Mean absolute error is:",mean_absolute_error(y_test, y_pred))


# **Analysis**<br>
# As we can see that the R2 Score of my model is positive,this shows that our model is performing well but the R2 score should be close to 1 for best performance,so we will try applying some other different models to improve our performance. 

# ### 2. DECISION TREE REGRESSOR:

# In[119]:


from sklearn import tree
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

D_T = tree.DecisionTreeRegressor(max_depth=8,random_state=42)
D_T.fit(X_train, y_train)
y_pred_dt = D_T.predict(X_test)
print('The Score on the training set with a decision tree regressor is:',D_T.score(X_train,y_train))

print('The Score on the test set with a decision tree regressor is:',D_T.score(X_test,y_test))
print("Root Mean squared error: %.2f"
      % np.sqrt(mean_squared_error(y_test, y_pred_dt)))
print("Mean squared error:"
      , mean_squared_error(y_test, y_pred_dt))

from sklearn.metrics import mean_squared_error
print("Mean absolute error is:",mean_absolute_error(y_test, y_pred_dt))

# The R^2 score 
print("The r2_score is: ", r2_score(y_test, y_pred_dt))


# **Analysis**<br>
# We see that by using the Decision regdression tree model the R2 score is coming out to be 0.82 which shows our model is very good.(if the R2 score of the model is near to 0 then the model is considered to be bad and a R2 score near to 1 is considered to be a good model)

# ### 3. KNN:

# In[120]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

neigh = KNeighborsRegressor(n_neighbors=15)
neigh.fit(X_train,y_train)
k_pred=neigh.predict(X_test)
print('The Score on the test set with a K neighbours regression is:', neigh.score(X_test,y_test))
print("Accuracy :",neigh.score(X_test,y_test)*100,'%')
from sklearn.metrics import mean_squared_error
print("Mean absolute error is:",mean_absolute_error(y_test, k_pred))
# The Root mean squared error
print("Root Mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test,k_pred )))
print("Mean squared error:"
      , mean_squared_error(y_test, k_pred))

# The R^2 score 
print("The r2_score is: ", r2_score(y_test, k_pred))


# **Analysis**<br>
# We see that by using the KNN model the R2 score is coming out to be ~0.78 which shows our model is fairly good.(if the R2 score of the model is near to 0 then the model is considered to be bad and a R2 score near to 1 is considered to be a good model)

# ### 4. RANDOM FOREST REGRESSOR:

# In[121]:


from sklearn.ensemble import RandomForestRegressor

rand_est = RandomForestRegressor()
rand_est.fit(X_train,y_train)
y_pred = rand_est.predict(X_test)
print('The Score on the test set with a random forest regressor is:', rand_est.score(X_test,y_test))
print("Root Mean squared error: %.2f"
      % np.sqrt(mean_squared_error(y_test, y_pred)))
print("Mean squared error:"
      , mean_squared_error(y_test, y_pred))
print("Accuracy :",rand_est.score(X_test,y_test)*100,'%')
print("The r2_score is: ", r2_score(y_test, y_pred))


# In[122]:


from sklearn.ensemble import RandomForestRegressor

rand_est = RandomForestRegressor(max_depth=20, n_estimators= 200, min_samples_split= 5, min_samples_leaf= 1, bootstrap= True, max_features= 'auto')
rand_est.fit(X_train,y_train)
y_pred_rfr = rand_est.predict(X_test)
print(' The Score on the train set with a hyperparameter optimized random forest regressor is:',rand_est.score(X_train,y_train))
print(' The Score on the test set with a hyperparameter optimized random forest regressor is:',rand_est.score(X_test,y_test))
from sklearn.metrics import mean_squared_error
print("Mean absolute error is:",mean_absolute_error(y_test, y_pred_rfr))
print("The r2_score is: ", r2_score(y_test, y_pred))


# **Analysis**<br>
# We see that by using the RANDOM FOREST REGRESSOR model the R2 score is coming out to be 0.91 which shows our model is really good.(if the R2 score of the model is near to 0 then the model is considered to be bad and a R2 score near to 1 is considered to be the best model)

# ### 5.Ridge Regression

# In[116]:


from sklearn.linear_model import Ridge
linridge=Ridge(alpha=20.0).fit(X_train,y_train)
y_pred_rr = linridge.predict(X_test)
print('Ridge Regression linear model intercept:{}'.format(linridge.intercept_))
print('Ridge Regression linear model coeff:{}'.format(linridge.coef_))
print('R squared Score(training):{:.3f}'.format(linridge.score(X_train,y_train)))
print('R squared Score(test):{:.3f}'.format(linridge.score(X_test,y_test)))
from sklearn.metrics import mean_squared_error
print("Mean absolute error is:",mean_absolute_error(y_test, y_pred_rr))
print("Root Mean squared error: %.2f"
      % np.sqrt(mean_squared_error(y_test, y_pred_rr)))
print("Accuracy :",linridge.score(X_test,y_test)*100,'%')


# In[111]:


pip install xgboost


# In[112]:


from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split,GridSearchCV


# In[113]:


xgb = XGBRegressor()

param_grid = {
    "booster": ['gbtree'],
    "eta": [0.01],
    "gamma": [0],
    "max_depth": [5],
    "lambda": [0],
    "alpha": [0]
}

xgb_model = GridSearchCV(estimator=xgb, cv=5, param_grid=param_grid)
xgb_model.fit(X_train, y_train)


# In[114]:


y_pred_train_xgb = xgb_model.predict(X_train)

print("MSE on training:", mean_squared_error(y_train, y_pred_train_xgb))

y_pred_valid_xgb = xgb_model.predict(X_test)

print("MSE on validation:", mean_squared_error(y_test, y_pred_valid_xgb))


# In[115]:


data.isnull().any().any()


# # PREDICTIONS:

# **The final dataset**

# In[177]:


X_test.head()


# # CONCLUSION:

# ### ->BEST MODEL

# We can compare all the models by checking their R2 score.<br>
# Condition: <br>
# * If R2 score is near to 1 -> Best model<br>
# * If R2 score is near to 0 -> Worst model<br>
# 
# **Models**<br>
# Linear Regression :-     0.55<br>
# Decison tree Regressor:-      0.82<br>
# K Nearest Neighbours(KNN):- 0.78<br>
# Random Forest Regressor:- 0.92<br>
# 
# So from observing the "R2 score" we conclude that *RANDOM FOREST REGRESSOR* is the best model for predicitng the price of the second hand car. 

# ### ->SUMMARY OF WHAT WE DID

# * Reading of the data<br>
# * Cleaning of the data by removing the null values for filling them with the median values. Also we considered the outliers present in the dataframe.<br>
# * Next was the EDA (Exploratory Data Analysis) in which we did a disciptive analysis of the entire dataframe with the help of mathematical values and visualization by checking the relation between the variables. <br>
# * Next we performed Label Encoding to deal with the categorical variables present.<br>
# * Next we have done the Scaling of the variable "Kilometers_Driven" to remove the units and make it fall in the range of 0 to 1 so as to avoid any favourable decisons by large or small values.<br>
# * Then we have splitted the dataframe into train and test cases to carry on the predictions.<br>
# * The Final step involved us using the different ML Algorithms so as to get the best model out of all for our price prediction.<br>

# In[ ]:




