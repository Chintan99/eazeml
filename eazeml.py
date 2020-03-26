#!/usr/bin/env python
# coding: utf-8

# ## eaze-ml   ¯\_(ツ)_/¯
# #### Data Science FrameWork

# In[25]:


## Basic Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
## interactive Visualization
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import cufflinks
cufflinks.go_offline(connected=True)
from plotly.offline import iplot# init_notebook_mode(connected=True)
import plotly.offline as pyo
pyo.init_notebook_mode()
#### Ignore Warning Messages
import os
import warnings
warnings.filterwarnings('ignore')

#### for Preprocessing
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,LabelBinarizer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.pipeline import make_pipeline


#### Machine Learning Essentials
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,GridSearchCV

#### Evaluation Metrices
from sklearn.utils.multiclass import unique_labels
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,log_loss,accuracy_score
from sklearn.metrics import mean_squared_error,r2_score,mean_squared_log_error


#### Machine Learning Models
### supervised
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor
from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMClassifier,LGBMRegressor
import lightgbm as lgb
from sklearn.svm import SVR,SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
### unsupervised
from sklearn.cluster import KMeans
### Neural Network
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

#### Feature Selection
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA


# In[6]:


### import data
def importdata(filename):
    '''
    Import Data from csv or excel in dataframe format
    example usage:
        dataframe = importtain('filename')
        just specify filename
    '''
    x = filename.split('.')[1]
    if x == 'xlsx':
        dataframe = pd.read_excel(filename)
    else:
        dataframe = pd.read_csv(filename)
    
    print('Dataframe Imported Successfully')
    return dataframe

def info(dataframe):
    '''
    ## Gives Entire Summary info of dataframe like nulls,uniques,
    percentage missing value,type of column of dataframe.
    ## gives you entire details of dataFrame
    '''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num = dataframe.select_dtypes(include=numerics).columns
    cat = dataframe.drop(num,axis=1)
    cat = cat.columns

    #cols = dataframe.columns
    #num = dataframe._get_numeric_data().columns
    #cat = list(set(cols) - set(num_cols))
    print('*** Dataset Infomarion ***')
    print('#####'*10)
    print ("No of Rows     : " ,dataframe.shape[0])
    print ("No of columns  : " ,dataframe.shape[1])
    print ('No of Numerical columns:',len(num))
    print ('No of Categorical columns:',len(cat))
    print('#####'*10)
    print ("\n Total Missing values :  ",dataframe.isnull().sum().values.sum(),'\n')
    print('#####'*10)
    types = []
    for i in dataframe.columns:
        if i in num:
            types.append('Numerical')
        else:
            types.append('Categorical')
    temp = pd.DataFrame()
    temp['Column Name'] = dataframe.columns
    temp['Nulls/NaN']= dataframe.isnull().sum().values
    temp['outof'] = dataframe.shape[0]
    temp['Unique'] = dataframe.nunique().values
    temp['Type of Columns'] = types
    print("Summary of DataFrame:\n\n",temp)
    if dataframe.isnull().sum().sum() > 0:
        missingdata(dataframe)

def missingdata(data):
    '''
    Plots Percentage missing value in barplot format
    '''
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    ms= ms[ms["Percent"] > 0]
    f,ax =plt.subplots(figsize=(8,6))
    plt.xticks(rotation='90')
    fig=sns.barplot(ms.index, ms["Percent"],color="green",alpha=0.8)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    plt.show()

def getcolumnstype(dataframe): 
    '''
   Returns categorical and numerical columns names from dataframe
   example usage:
           num, cat = getcolumnstype(dataframe)
    '''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num = dataframe.select_dtypes(include=numerics).columns
    cat = dataframe.drop(num,axis=1)
    cat = cat.columns
    return num,cat

def detail_cat(dataframe):
    '''
    Prints detial of categoy columns
    '''
    num, cat = getcolumnstype(dataframe)
    print('Deatils of Categorical Columns:\n\n')
    for i in cat:
        print(i,">> column has values :\n",dataframe[i].value_counts())
        print("* * "*10)

## for dropping columns just name as maany as columns 
def dropcolumns(dataframe,*argv):
    '''
    Drops columns specified from dataframe and return dataframe
    example usage:
        dataframe = dropcolumns(dataframe,'columnam')
        specify column names in '' seprated by ','comma
    '''
    columns = list(argv)
    dataframe.drop(columns,axis=1,inplace=True)
    return dataframe
# -------- advance --------------
def drop_columns(dataframe,n=0,*argv):
    '''
    Drop columns which have null value percentage above specified in n
    example usage:
            dataframe = drop_columns(dataframe,n,column names)
            n = set to 0 if don't wan't to use percenatage missing feature
            example drop_columns(df,0,'id')
    '''
    if n > 0:
        data = dataframe.copy()
        total = data.isnull().sum().sort_values(ascending = False)
        percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
        ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        ms= ms[ms["Percent"] > 0]
        print("Null value summary: \n",ms)
        dropout = percent[percent > n].index
        print('columns dropped',dropout)
        data.drop(dropout,axis=1,inplace=True)
        
        columns = list(argv)
        data.drop(columns,axis=1,inplace=True)
        print('\n\ncolumns dropped \n\n',list(dropout),"\n",columns)
        return data
    else:
        columns = list(argv)
        dataframe.drop(columns,axis=1,inplace=True)
        print("\n\n",columns,"> Dropped \n\n")
        return dataframe

def get_IQR(dataframe):
    """
    Prints IQR table for dataframe
    """
    Q1 = dataframe.quantile(0.25)
    Q3 = dataframe.quantile(0.75)
    IQR = Q3-Q1
    print(IQR)

def show_outlier(df):
    '''
    show no of outliers in each column
    '''
    for i in df._get_numeric_data().columns:
        low,high = df[i].quantile([0.1,0.95]).values
        rows = df[(df[i] > low) & (df[i] < high)].shape[0]
        if rows > 0:
            print(rows,'  No of outliers in column >>>',i)
        else:
            print('***** no outlier in ',i)

def remove_outlier(dataframe):
    '''
    Removes all outlier data from dataframe from entire dataframe
    example usage:
        newdf = remove_outlier(dataframe)
        # newdf will have no outlier values
    '''
    Q1 = dataframe.quantile(0.5)
    Q3 = dataframe.quantile(0.95)
    IQR = Q3-Q1
    newdataframe = dataframe[~((dataframe < (Q1 - 1.5 * IQR)) |(dataframe > (Q3 + 1.5 * IQR))).any(axis=1)]
    print('IQR Range for all columns\n',IQR)
    return newdataframe

def fillnulls(dataframe,value,*argv):
    '''
    Fill null values in dataframe columns with specified value or with mean,median,mode
    example usage:
            train = fillnulls(train,'median','batch_enrolled'...)
            fills median value of respctive column in place of null values
            
        if wan't to specify any other values beside mean,median,mode
        just specigy that value:
            train = fillnulls(train,'value','batch_enrolled'...)
    '''
    dataframe1 = dataframe.copy()
    columns = list(argv)
    print('\nColumns to be cleaned are :\n',columns,'\n')
    if value == 'mean':
        for i in columns:
            val = dataframe1[i].mean()
            dataframe1[i] = dataframe1[i].fillna(val)
            print(i,'filled with:',val,'\n')
        return dataframe1
    elif value == 'median':
        for i in columns:
            val = dataframe1[i].median()
            dataframe1[i] = dataframe1[i].fillna(val)
            print(i,'filled with:',val,'\n')
        return dataframe1
    elif value == 'mode':
        for i in columns:
            val = dataframe1[i].mode
            dataframe1[i] = dataframe1[i].fillna(val)
            print(i,'filled with:',val,'\n')
        return dataframe1
    else:
        for i in columns:
            dataframe1[i] = dataframe1[i].fillna(value)
            print("\n",value,'Filled inplace of NaNs\n')
        return dataframe1

def extract_number(dataframe,columname):
    '''
    extract numbers from column 
    example usage:
            dataframe = extract_number(dataframe,'columname')
        just spcify dataframe and columname from which no to be extracted
        and it return cleaned column
        Null values will be replaced with 9999
    '''
    if dataframe[columname].isnull().sum() > 0:
        dataframe[columname].fillna('9999',inplace=True)
    
    dataframe[columname] = dataframe[columname].astype(str)
    dataframe[columname].replace("[^0-9]","",regex=True,inplace=True)
    dataframe[columname] = dataframe[columname].apply(lambda x: x.strip())
    return dataframe

def label_encode(dataframe):
    '''
    Perfoms label Encoding on all categorical features and return one label encoded data
    usage example :
            dataframe = onehot_encode(dataframe)
    '''
    data = dataframe.copy()
    num,cat = getcolumnstype(data)
    print('\ncolumns label encoded\n',cat)
    data[cat] = data[cat].apply(LabelEncoder().fit_transform)
    return data
    
def onehot_encode(dataframe):
    '''
    Perfoms One Hot Encoding on all categorical features and return one hot encoded data
    usage example :
            dataframe = onehot_encode(dataframe)
    '''
    data = dataframe.copy()
    num,cat = getcolumnstype(data)
    print('\ncolumns onehot encoded\n',cat)
    #data[cat] = data[cat].apply(OneHotEncoder().fit_transform)
    data = pd.get_dummies(data,drop_first=True)
    return data

def rfe(model,X_train,y_train,n=10):
    '''
    Performs Recurssive Feature Elimination and return best n feaures specified (defualt set to 10)
    example usage:
            rfecolumns = rfe(model,X_train,y_train,n)
        model - machine learning algorithm model
        X_train,y_train - generated from train_test_split
        n - (optional) set no of features to be selected
    '''
    rfe = RFE(model, n)
    print('\nExecuting RFE\nPlease wait.........')
    X_train = scale(X_train)
    rfe.fit(X_train,y_train)
    print('### RFE selected columns:\n',X_train.columns[rfe.support_])
    return X_train.columns[rfe.support_]

def scale(dataframe):
    mm = MinMaxScaler()
    c = mm.fit_transform(dataframe)
    df = pd.DataFrame(c,columns = dataframe.columns)
    return df

def VIF(dataframe):
    '''
    Prints Variance Inflation Factor value for each column in dataframe in tabluar format
    '''
    vif = pd.DataFrame()
    vif['feature'] = dataframe.columns
    vif['vif'] = [variance_inflation_factor(dataframe.values,i)for i in range(dataframe.shape[1])]
    vif['vif'] = round(vif['vif'],2)
    vif = vif.sort_values(by='vif',ascending=False)
    return print('\n\n******** VIF *********\n\n',vif)

def stat_models(y, X):
    '''
    Display statastical table for X and Y
    '''
    model = sm.OLS(y, X)
    results = model.fit()
    print("***"*20)
    print("\n\n",results.summary())
    
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

def classification_result(y_test,y_pred):
    '''
    Evaluate Performace with Popular Metrices:
    gives different scores when passed y_true & y_pred
    example usage:
            regression_result(y_true,y_pred)
            
    '''
    print("--------------------------------------------------------")
    print('********************* Model Results ********************')
    print("--------------------------------------------------------")
    print("F1 Score :",f1_score(y_test,y_pred,average = "weighted"))
    print('AUC-ROC Score :',multiclass_roc_auc_score(y_test, y_pred)) ## only for Binary Classification
    print('Report:\n',classification_report(y_test, y_pred))
    confusion_mat(y_test,y_pred)
    if len(y_test.value_counts()) == 2:
        roc_curve_graph(y_test, y_pred)
    print("--------------------------------------------------------")

def regression_result(y_test,y_pred):
    '''
    Evaluate Performace with Popular Metrices:
    gives different scores when passed y_true & y_pred
    example usage:
            regression_result(y_true,y_pred)
            
    '''
    print("-----------------------------------------")
    print('************* Model Results *************')
    print("-----------------------------------------")
    print('R2 score:',r2_score(y_test,y_pred))
    print('MSE score:',mean_squared_error(y_test, y_pred))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('RMSE score:',rmse)
    print('RMSLE score:',(mean_squared_log_error(y_test,abs(y_pred))))
    print("-----------------------------------------")

def cross_validate(model,x,y,n=3):
    '''
    Returns cross validation score 
    example usage:
            cross_validate
    '''
    score = cross_val_score(model, X, y, scoring='accuracy',n_jobs=-1, cv=n)
    cv_score = (round(np.mean(score),5) * 100)
    print('Cross-validation Score >>',cv_score)

def roc_curve_graph(y_test, y_pred):
    '''
    Plots Area Under Curve graph 
    #### only works for binary classification
    example usage:
            roc_curve_graph(actual-values,predicted-values)
    '''
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

#### Basic Machine learning Models
def linear_reg(X_train,y_train,X_test,y_test):
    '''
    performs Logistic Regression Algorithm and returns predicted values and model
    usage example:
            y_pred, model = lgb_reg(X_train,y_train,X_test,y_test)
    '''
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    return y_pred, lr

def logistic_reg(X_train,y_train,X_test,y_test):
    '''
    performs Linear Regression Algorithm and returns predicted values and model
    usage example:
            y_pred, model = lgb_reg(X_train,y_train,X_test,y_test)
    '''
    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)
    return y_pred, lr

def randomforest_classifier(X_train,y_train,X_test,y_test):
    '''
    Performs Random Forest Algorithm and returns predicted values and model and plot feature importance graph
    usage example:
            y_pred, model = lgb_reg(X_train,y_train,X_test,y_test)
    '''
    rfc = RandomForestClassifier(n_estimators = 200,random_state=101,class_weight='balanced')
    rfc.fit(X_train,y_train)
    y_pred = rfc.predict(X_test)
    try:
        #### plotting feature importance
        importances=rfc.feature_importances_
        feature_importances=pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
        plt.figure(figsize=(10,7))
        sns.barplot(x=feature_importances[0:10], y=feature_importances.index[0:10])
        plt.title('Random Forest Feature Importance',size=20)
        plt.ylabel("Features")
        plt.show()
    except:
        print('Unable to plot Feature Importance Graph')
    return y_pred,rfc

def randomforest_reg(X_train,y_train,X_test,y_test):
    '''
    Performs Random Forest Algorithm and returns predicted values and model and plot feature importance graph
    usage example:
            y_pred, model = lgb_reg(X_train,y_train,X_test,y_test)
    '''
    rfc = RandomForestRegressor(n_estimators=100,max_depth=8, random_state=101)
    rfc.fit(X_train,y_train)
    y_pred = rfc.predict(X_test)
    try:
        #### plotting feature importance
        importances=rfc.feature_importances_
        feature_importances=pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
        plt.figure(figsize=(10,7))
        sns.barplot(x=feature_importances[0:10], y=feature_importances.index[0:10])
        plt.title('Random Forest Feature Importance',size=20)
        plt.ylabel("Features")
        plt.show()
    except:
        print('Unable to plot Feature Importance Graph')
    return y_pred, rfc

def xgb_classifier(X_train,y_train,X_test,y_test):
    '''
    Performs XGBoost Algorithm and returns predicted values and model and plot feature importance graph
    usage example:
            y_pred, model = xgb_classifier(X_train,y_train,X_test,y_test)
    '''
    xgb = XGBClassifier(n_estimator=100,max_depth=8,class_weight='balanced',refit='AUC')
    xgb.fit(X_train,y_train)
    y_pred = xgb.predict(X_test)
    return y_pred, xgb

def xgb_reg(X_train,y_train,X_test,y_test):
    '''
    Performs XGBoost Algorithm and returns predicted values and model and plot feature importance graph
    usage example:
            y_pred, model = xgb_reg(X_train,y_train,X_test,y_test)
    '''
    model_xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =42, nthread = -1)
    xgb.fit(X_train,y_train)
    y_pred = xgb.predict(X_test)
    return y_pred, model_xgb

def lgb_classifier(X_train,y_train,X_test,y_test):
    '''
    Performs LGBM boost Algorithm and returns predicted values and model and plot feature importance graph
    usage example:
            y_pred, model = lgb_classifier(X_train,y_train,X_test,y_test)
    '''
    y_train1 = y_train.values
    model = lgb.LGBMClassifier(n_estimators=600,random_state=101,max_depth=8,
                               verbose=1,class_weight='balanced')
    model.fit(X_train, y_train1)
    y_pred = model.predict(X_test)
    try:
        #### plotting feature importance
        fig, ax = plt.subplots(figsize=(12,8))
        lgb.plot_importance(model, max_num_features=10, height=0.8, ax=ax)
        ax.grid(False)
        plt.title("LightGBM - Feature Importance", fontsize=15)
        plt.show()
    except:
        print('Unable to plot Feature Importance Graph')
    return y_pred, model

def lgb_reg(X_train,y_train,X_test,y_test):
    '''
    Performs LGBM boost Algorithm and returns predicted values and model and plot feature importance graph
    usage example:
            y_pred, model = lgb_reg(X_train,y_train,X_test,y_test)
    '''
    model = lgb.LGBMRegressor(n_estimators=600,random_state=101,max_depth=8,
                               verbose=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    try:
        #### plotting feature importance
        fig, ax = plt.subplots(figsize=(12,8))
        lgb.plot_importance(model, max_num_features=10, height=0.8, ax=ax)
        ax.grid(False)
        plt.title("LightGBM - Feature Importance", fontsize=15)
        plt.show()
    except:
        print('Unable to plot Feature Importance Graph')
    return y_pred, model

def adaboost_classifier(X_train,y_train,X_test,y_test):
    '''
    Performs Ada Boost Algorithm and returns predicted values and model and plot feature importance graph
    usage example:
            y_pred, model = adaboost_classifier(X_train,y_train,X_test,y_test)
    '''
    ada = AdaBoostClassifier(n_estimators=100)
    ada.fit(X_train,y_train)
    y_pred = ada.predict(X_test)
    try:
        importances=ada.feature_importances_
        feature_importances=pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
        plt.figure(figsize=(10,7))
        sns.barplot(x=feature_importances[0:10], y=feature_importances.index[0:10])
        plt.title('AdaBoost Feature Importance',size=20)
        plt.ylabel("Features")
        plt.show()
    except:
        print('Unable to plot Feature Importance Graph')
    return y_pred, ada

def adaboost_reg(X_train,y_train,X_test,y_test):
    '''
    Performs Ada Boost Algorithm and returns predicted values and model and plot feature importance graph
    usage example:
            y_pred, model = adaboost_reg(X_train,y_train,X_test,y_test)
    '''
    ada = AdaBoostRegressor(n_estimators=100, random_state=0)
    ada.fit(X_train,y_train)
    y_pred = ada.predict(X_test)
    try:
        importances=ada.feature_importances_
        feature_importances=pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
        plt.figure(figsize=(10,7))
        sns.barplot(x=feature_importances[0:10], y=feature_importances.index[0:10])
        plt.title('AdaBoost Feature Importance',size=20)
        plt.ylabel("Features")
        plt.show()
    except:
        print('Unable to plot Feature Importance Graph')
    return y_pred, ada

def Gradient_boost_classifier(X_train,y_train,X_test,y_test):
    '''
    Performs Gradient Boost Algorhtm and returns predicted values and model
    usage example:
            y_pred, model = Gradient_boost_reg(X_train,y_train,X_test,y_test)
    '''
    GBoost = GradientBoostingClassifier(n_estimators = 3000, max_depth = 5,max_features='sqrt',
                                            min_samples_split = 10,learning_rate = 0.005,
                                            min_samples_leaf=15,random_state =10)
    GBoost.fit(X_train, y_train)
    y_pred = GBoost.predict(X_test)
    return y_pred, GBoost

def Gradient_boost_reg(X_train,y_train,X_test,y_test):
    '''
    Performs Gradient Boost Algorhtm and returns predicted values and model
    usage example:
            y_pred, model = Gradient_boost_reg(X_train,y_train,X_test,y_test)
    '''
    GBoost = GradientBoostingRegressor(n_estimators = 3000, max_depth = 5,max_features='sqrt',
                                            min_samples_split = 10,learning_rate = 0.005,loss = 'huber',
                                            min_samples_leaf=15,random_state =10)
    GBoost.fit(X_train, y_train)
    y_pred = GBoost.predict(X_test)
    return y_pred, GBoost

def grid_search(model,flag,X_train,y_train,X_test,y_test):
    '''
    Performs GridSearchCV when model and parameters passed and return predicted values and best_estimaor parameters 
    usage example:
            y_pred, model = grid_search(model,flag,x_train,y_train,X_test,y_test)
        model - pass any machine learning model
        flag: 
            1. 'quick' (get quicky executed)
            2. 'deep' (takes time)
        X_train,y_train,X_test,y_test generated from train_test_split
        
    '''
    if flag == 'quick':
        params = {
            #'bootstrap': [True],
            'max_depth': [80, 90, 100, 110],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [8, 10, 12],
            'n_estimators': [20, 30, 40, 50],
        }
    elif flag == 'deep':
        params = {'min_child_weight':[5,6,7], 'gamma':[i/10.0 for i in range(3,7)],
                  'subsample':[i/10.0 for i in range(6,11)],
                  'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [5,6,7,8]}
    
    print('Performing GridSearchCV \nPlease wait............')
    grid = GridSearchCV(model, params)
    grid.fit(X_train,y_train)
    y_pred = grid.best_estimator_.predict(X_test)
    print('Done  ¯\_(ツ)_/¯\n')
    print('best parameters: \n', grid.best_estimator_)
    return y_pred, grid

def kmeans(dataframe,n):
    '''
    Performs Kmeans clustering Algorithm
    return preidicted clusters when passed data and no of required clusters
    example usage:
            clusters = (dataframe,n)
        n- no of required clusters
    '''
    km = KMeans(n_clusters=n)
    y_pred_c = km.fit_predict(dataframe)
    return y_pred_c 

def classification(X,y):
    '''
This Method Provides Menu based Algorithm Selection (Classification) for Prediction,It return predicted values(y_pred) and machine learning model for further usage.
    example usage:
             y_pred, model = classification(X,y)
    X - features
    y - target
    ### just pass cleaned X and y
    '''
    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.7,random_state=42)
    print("########## MENU ##############\n")
    print('1. Logisitic Regression \n2. Random Forest \n3. XGB Classifier \n4. LGB CLassifier \n5. Gradient Boosting Classifier \n6. AdaBoost Classifier\n')
    x = int(input('Input No, which you want:'))
   # if x == 99:
     #   return quick_pred(X,y,'c')
    if x == 1:
        print('Predicting with Logisitic Regression Classifier')
        y_pred, model = logistic_reg(X_train,y_train,X_test,y_test)
    elif x == 2:
        print('Predicting with Random Forest Classifier')
        y_pred, model = randomforest_classifier(X_train,y_train,X_test,y_test)
    elif x == 3:
        print('Predicting with XGB boost Classifier')
        y_pred, model = xgb_classifier(X_train,y_train,X_test,y_test)
    elif x == 4:
        print('Predicting with Light GBM Classifier')
        y_pred, model = lgb_classifier(X_train,y_train,X_test,y_test)
    elif x == 5:
        print('Predicting with Gradient Bossting Classifier')
        y_pred, model = Gradient_boost_classifier(X_train,y_train,X_test,y_test)
    elif x == 6:
        print('Predicting with Ada Boost CLassifier')
        y_pred, model = adaboost_classifier(X_train,y_train,X_test,y_test)
    else:
        print('\n\n Please input appropirate no (*-*)')
        classification(X_train,y_train,X_test,y_test)
    
    classification_result(y_test,y_pred)
    print('\nModel used:',model,'\n Done.')
    return y_pred,model

def regression(X,y):
    '''
This Method Provides Menu based Algorithm Selection (Regression) for Prediction, It return predicted values(y_pred) and machine learning model for further usage.
    example usage:
             y_pred, model = classification(X,y)
    X - features
    y - target 
    ### just pass cleaned X and y
    '''
    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.7,random_state=42)
    print("########## MENU ##############\n")
    print('1. linear Regression \n2. Random Forest regressor \n3. XGB regressor \n4. LGB regressor \n5. Gradient Boosting regressor \n6.AdaBoost regressor\n')
    x = int(input('Input No, which you want:'))
    #if x == 99:
     #   return quick_pred(X,y,'r')
    if x == 1:
        print('Predicting with linear Regression')
        y_pred, model = linear_reg(X_train,y_train,X_test,y_test)
    elif x == 2:
        print('Predicting with Random Forest Regressor')
        y_pred, model = randomforest_reg(X_train,y_train,X_test,y_test)
    elif x == 3:
        print('Predicting with XGB boost Regressor')
        y_pred, model = xgb_reg(X_train,y_train,X_test,y_test)
    elif x == 4:
        print('Predicting with Light GBM Regressor')
        y_pred, model = lgb_reg(X_train,y_train,X_test,y_test)
    elif x == 5:
        print('Predicting with Gradient Bossting Regressor')
        y_pred, model = Gradient_boost_reg(X_train,y_train,X_test,y_test)
    elif x == 6:
        print('Predicting with Ada Boost Regressor')
        y_pred, model = adaboost_classifier(X_train,y_train,X_test,y_test)
    else:
        print('\n\n Please input appropirate no (*-*)')
        regression(X_train,y_train,X_test,y_test)
    
    regression_result(y_test,y_pred)
    print('\nModel used:',model,'\n Done.')
    return y_pred,model


# ## Quick Machine learning

# In[7]:


### please to input only cleaned dataframe
### specify "r" for regression and 'c' for classification
### X are features, y is target
def clean_data(dataframe):
    dataframe.apply(lambda x: x.replace(" ",np.nan,inplace=True))
    dataframe = drop_columns(dataframe,20)
    nullcolumns = []
    for i,j in dict(dataframe.isnull().sum()).items():
        if j > 0:
            nullcolumns.append(i)
    
    num1, cat1 = getcolumnstype(dataframe[nullcolumns])
    dataframe = fillnulls(dataframe,'unknown',cat1)
    dataframe = fillnulls(dataframe,'median',num1)
    return dataframe

def quick_pred(X,y,flag):
    '''
    Automatically Trains & Tests data using Many Famous Machine learning Algrithms and return scores in colorized dataframe table format.
    **** Passed data should be cleaned *****
    #### use quick_ml() function for uncleaned data 
example:
        quick(X,y,flag)
    * X: Specify features
    * y: spcifigy Target column
    * flag: specify 'r'for regression & 'c' for classification
    '''
    #ss = StandardScaler()
    #X = ss.fit_transform(X)
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=101,test_size=30)
    print('\nPlease wait Training-Testing with all models..\n')
    if flag.lower() == 'r':
        models = {"LinearRegression": LinearRegression(),
                  "RandomForestRegression": RandomForestRegressor(),
                  "XGBoostRegressor":XGBRegressor(),
                  "LGBoostRegressor":LGBMRegressor(),
                  "AdaBoostRegressor":AdaBoostRegressor(),
                  "SupportVectorMachine":SVR(),
                  "GradientBoostingRegression": GradientBoostingRegressor()}
        mn,r2score,rmse,rmsle,mse = [],[],[],[],[]

        for model_name, model in tqdm(models.items()):
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mn.append(model_name)
            r2score.append(r2_score(y_test,pred))
            rmse.append(np.sqrt(mean_squared_error(y_test,pred)))
            rmsle.append(np.sqrt(mean_squared_log_error(y_test,abs(pred))))
            mse.append(mean_squared_error(y_test,pred))
            print('Done with ', model_name)
        
        results = pd.DataFrame()
        results['Model'] = mn
        results['R^2 score'] = r2score
        results['Root Mean Squared Error'] = rmse
        results['Root Mean Squared Log Error'] = rmsle
        results['Mean Squared Error'] = mse
        results = results.sort_values(by=['R^2 score'],ascending=False)
        return results.style.background_gradient()
    
    elif flag.lower() == 'c':
        models = {"LogisticRegression": LogisticRegression(),
                      "RandomForestClassifier": RandomForestClassifier(),
                      "XGBoost Classifier":XGBClassifier(),
                      "LGBoost Classifier":LGBMClassifier(),
                      "AdaBoost Classifier":AdaBoostClassifier(),
                      #"SupportVectorMachine":SVC(),
                      "GradientBoostingClassifier": GradientBoostingClassifier()}
        f1,auc,mn,acc,ll = [],[],[],[],[]
        for model_name, model in tqdm(models.items()):
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                mn.append(model_name)
                f1.append(f1_score(y_test,pred,average = "weighted"))
                if len(pd.Series(pred).value_counts()) == 2:
                    auc.append(roc_auc_score(y_test,pred))
                else:
                    auc.append(multiclass_roc_auc_score(y_test, pred))
                acc.append(accuracy_score(y_test, pred))
                ll.append(balanced_accuracy_score(y_test,pred))
                print('Done with ', model_name)

        results = pd.DataFrame()
        results['Model'] = mn
        results['F1 score'] = f1
        results['AUC-ROC score'] = auc
        results['Accuracy'] = acc
        results['Balanced Accuracy Score'] = ll
        results = results.sort_values(by=['F1 score'],ascending=False)
        print(results)
        print('\nCheck returned table\n')
        return results.style.background_gradient()
    

### pass Dataframe,Target column-name & flag (r for regression and c for classification)
def quick_ml(df1,target,flag,n=0):
    '''
quick_ml: No hazzle Machine learning.
    This methods Automatically takes dataframe and target column and flag as input and gives you the Preidiction using famous Machine learning Algorithm.
example usage:- 
            result, model = quick_ml(dataframe,'target-column-name','flag')
parameters:
 * dataframe: the dataframe you wan't to get prediction from
 * 'target-column-name' : specify target column name
 * 'flag': specify flag 'r'for regression and 'c'for classification
optional:
  'n'= no of features you wan't to be used for prediction using RFE
      (by default set to 0 and predicts using all features )
      (Rfe May Take Time in Execution so wait)
    
    '''
    info(df1)
    if df1.isnull().sum().sum() > 0:
        df1 = clean_data(df1)
        
    num,cat = getcolumnstype(df1)
    if len(cat) > 0:
        df1=label_encode(df1)
    corr_heatmap(df1,'interactive')
    X = df1.drop(target,axis=1)
    y = df1[target]
    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.7,random_state=42)

    if flag == 'c':
        showbias(df1,target)
        print('Detailed Executing with RandomForest ..')
        y_pred, model = randomforest_classifier(X_train,y_train,X_test,y_test)
        classification_result(y_test,y_pred)
        if n>0:
            print('\n please wait while performing prediction using RFE...')
            rfecols = rfe(model,X_train,y_train,n)
            X_train = X_train[rfecols]
            X_test = X_test[rfecols]
            print('RFE Selected Features',rfecols)
            y_pred, model = randomforest_classifier(X_train,y_train,X_test,y_test)
        X = scale(X)
        return quick_pred(X,y,'c'),model
    
    elif flag == 'r':
        box_hist_plot(df1,target)
        print('Detailed Executing with RandomForest ..')
        y_pred, model = randomforest_reg(X_train,y_train,X_test,y_test)
        regression_result(y_test,y_pred)
        if n>0:
            print('\n please wait while performing prediction using RFE...')
            rfecols = rfe(model,X_train,y_train,n)
            X_train = X_train[rfecols]
            X_test = X_test[rfecols]
            y_pred, model = randomforest_reg(X_train,y_train,X_test,y_test)
            print(' RFE Selected Features')
        X = scale(X)
        return quick_pred(X,y,'r'),model


# ## Visualization framework
# * showbias()
# * corr_heatmap()
# * box_hist_plot()

# In[8]:


def showbias(dataframe,target):
    '''
    Plots interactive Pie Graph for Column and show percentage of each class in column
    example usage:-
      showbias(dataframe,'columname')
    '''
    data = dataframe.copy()
    labels = list(data[target].value_counts().index)
    fig = px.pie(names = labels,values = data[target].value_counts().values,title='Percentage data of '+target)
    fig.update_traces(hole=.4, hoverinfo="label+percent+name")
    fig.show()

def corr_heatmap(dataframe,style=None):
    ''' 
    Plot interactive corealtion heatmap for dataframe passed.
    example usage:- 
      corr_heatmap(dataframe,style)
      optional style:
        1. 'basic' - return basic tabular highlighted corelation table 
        2. interactive - plots interactive(zoomable) heatmap with plotly 
    
    if columns names are not passed it will plot histrogram and boxplot for all numeric columns
    '''
    if style == 'basic':
        return dataframe.corr().style.background_gradient()
    elif style =='interactive':
        corrs = dataframe.corr()
        figure = ff.create_annotated_heatmap(
            z=corrs.values,
            x=list(corrs.columns),
            y=list(corrs.index),
            annotation_text=corrs.round(2).values,
            showscale=True)
        figure.show()
    else:
        plt.figure(figsize=(20,20))
        sns.heatmap(dataframe.corr(),square=True,annot=True,linewidths=0.2,cmap='Greens')

def box_hist_plot(dataframe,*argv):
    ''' 
    Plot interactive Boxplot & Histogram for n no of column passed after dataframe name.
    example usage:- 
        box_hist_plot(dataframe,'columname1','columname2')
    
    if columns names are not passed it will plot histrogram and boxplot for all numeric columns
    '''
    columns = list(argv)
    if len(columns) > 0:
        for i in columns:
            dataframe[i].iplot(kind='hist', title= i+' Distribution')
            dataframe[i].iplot(kind='box', title= i+' Box plot')
    else:
        for i in dataframe._get_numeric_data().columns:
            dataframe[i].iplot(kind='hist', title= i+' Distribution')
            dataframe[i].iplot(kind='box', title= i+' Box plot')
            
def confusion_mat(y_test,y_pred,cmap=None):
    '''
    plots confusion matrix
    '''
    if cmap != None:
        clr = cmap
    else:
        clr = 'gnuplot2_r'
    y_pred_temp = pd.Series(y_pred)
    labels = unique_labels(y_test,y_pred)
    sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,
                fmt='d',cmap=clr,
                cbar=False,
                xticklabels = labels,
                yticklabels = labels,
                linewidth=0.2,
                linecolor='black'
               )
    plt.ylabel('Actual',size=15)
    plt.xlabel('Predicted',size=15)
    plt.title('Confusion Matrix',size=20)
    plt.show()

from wordcloud import WordCloud 
def wordcloud(df1,columname,bgcolor=None):
    '''
    df : dataframe
    columname : columne for which word cloud needed
    bgcolor : black or white background
    '''
    if bgcolor != None:
        bgcolor = bgcolor
    from wordcloud import WordCloud 
    x2011 = df1[columname]
    plt.subplots(figsize=(8,8))
    wordcloud = WordCloud(
                              background_color=bgcolor,
                              width=512,
                              height=384
                             ).generate(" ".join(x2011))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


# ### Kaggle 

# In[9]:


def kaggle(traindataset,testdataset,target,flag):
    train = traindataset.copy()
    test = testdataset.copy()
    '''
    Kaggle Methods Makes Particiapting in Kaggle Compitions easy
        - train = traindataframe
        - test = testdataframe
        - target = 'target column name'
        - flag = 'Specify operation type 
                ('r' for regression & 'c' for classification)
    '''
    print('\n\n******* Train Dataset Information *********\n')
    info(train)
    print('\n\n********* Test Dataset Information ***********\n')
    info(test)
    target_var = train[target]
    train.drop(target,inplace=True,axis=1)
    if train.isnull().sum().sum() > 0:
        train = clean_data(train)
    
    temp = list(train.columns)
    test = test[temp]
    if test.isnull().sum().sum() > 0:
        nullcolumns = []
        for i,j in dict(train.isnull().sum()).items():
            if j > 0:
                nullcolumns.append(i)
        test = fillnulls(test,'median',nullcolumns)
    
    newdf = train.append(test)
    num,cat = getcolumnstype(newdf)
    if len(cat) > 0:
        newdf = label_encode(newdf)
    train = newdf[:train.shape[0]]
    test = newdf.tail(test.shape[0])
    train[target] = target_var
    corr_heatmap(train,'interactive')
    X = train.drop(target,axis=1)
    y = train[target]
    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.7,random_state=42)
    if flag == 'c':
        showbias(train,target)
        y_pred,model = classification(X,y)
        report = quick_pred(X,y,'c')
        print(report)
    if flag == 'r':
        box_hist_plot(train,target)
        y_pred,model = regression(X,y)
        report = quick_pred(X,y,'r')
        print(report)
    
    test_pred = model.predict(test)
    return test_pred,report

def kaggle_csv(column1,column2,filename,*argv):
    '''
    Method inputs column1,column2 and filename and makes csv of that filename.
    you can specify columnname after filename else default name would be id and target.
    '''
    if len(argv) > 1:
        index = argv[0]
        target = argv[1]
    else:
        index = 'id'
        target = 'target'
    subdf = pd.DataFrame()
    subdf[index] = column1
    subdf[target] = column2
    nfn = filename+".csv"
    subdf.to_csv(nfn, index=False)
    print('CSV created Successfully named: ',nfn)
    return subdf


# In[10]:


import eazeml
import inspect


# In[11]:


def help(fn):
    print(fn.__doc__)
def list_func():
    method_list = [ func[0] for func in inspect.getmembers(eazeml, predicate=inspect.isroutine) if callable(getattr(eazeml, func[0]))]
    print('List of Available Methods :\n',)
    for i in method_list:
        print(i)


# ### NLP processing 

# In[12]:


from textblob import TextBlob ### For Sentiment Polarity
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import string
import nltk
from nltk.stem import PorterStemmer,WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.notebook import tqdm

def text_process(mess):
    '''
    Method removes Punctuations and Special Characters from column
    and return lower cased
    '''
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    nopunc =  ' '.join([word for word in nopunc.split() if word.lower() not in stop_words])
    return str(TextBlob(nopunc)).lower()

def nlp_text(dataframe,columname):
    '''
    Method converts Textual data of Column to Tfidf vector after:
    Removing Punctutions, Speical Symbol/character, Coverts Uppercase to lowercase and Lemmetization
    ### only column passed is converted to vector not whole dataframe
    ### need to manually join output data vector with original dataframe
    example usage:
            data = nlp_text(dataframe,'columname')
    ## after this execute this :
        dataframe.drop('columname',axis=1)
        newdf = pd.concat([dataframe,pd.DataFrame(data)],axis=1)
    '''
    joined = dataframe[columname]
    joined.replace("[^a-zA-Z]"," ",regex=True,inplace=True)
    joined = joined.apply(text_process)

    story = []
    for i in joined:
        story.append(i)

    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    new_sentence = []
    for i in tqdm(range(len(story))):
        words = nltk.word_tokenize(story[i])
        words = [lemmatizer.lemmatize(word) for word in words]
        new_sentence.append(' '.join(words))

        ## just checking result with both, Using Lemmatization as it is good.
        #for word in words:
            #print(word,"---to setm-------->",stemmer.stem(word))
            #print(word,'---to lematize---->',lemmatizer.lemmatize(word))
    print('Processed Text looks like >>\n',new_sentence[2])

    tfid = TfidfVectorizer() 
    data = tfid.fit_transform(new_sentence).toarray()
    print('\nshape before Text Processing:',dataframe[columname].shape)
    print('\nshape before Text Processing:',data.shape)
    return data


# ### Deep Learning

# In[14]:


from sklearn.neural_network import MLPClassifier, MLPRegressor
def neural_network(X_train,y_train,X_test,y_test,flag,solver=None):
    '''
    example usage:
        y_pred,model = neural_network(X_train,y_train,X_test,y_test,flag)
        flag: 'r' for regresssion ,'c' for classification 
        optional solver: 'adam','lbfgs','sgd'
        
    Neural Netowrks needs to be hardcoded for every dataset
    try optimizing them:
        solver = 'adam','lbfgs','sgd'
        hidden_layer_sizes=(100, 100),
        solver='adam',tol=1e-2, 
        max_iter=500, random_state=1
    '''
    print('Please wait, Building Neural Netowrk....')
    if solver != None:
        bp = solver
    else:
        bp = 'lbfgs'
    
    if flag == 'c':
        mlp = make_pipeline(StandardScaler(),
                            MLPClassifier(solver = bp))
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        classification_result(y_test,y_pred)
    elif flag =='r':
        #hidden_layer_sizes=(100, 100),solver='adam',tol=1e-2, max_iter=500, random_state=1
        mlp = make_pipeline(StandardScaler(),
                            MLPRegressor(solver = bp))
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        regression_result(y_test,y_pred)
    else:
        print('Please Specify Flag (r) or (c)')
    
    return y_pred, mlp


# In[15]:


def gen_txt_report(var):
    '''
    usage:
    1. write '%%captuer example' at top of cell you want to make report of
    2. use gen_report(example)
    example:
        %%capture example
        quick_ml(df,'target','c')
        gen_report(example) ## this creates summary file named output.txt
    '''
    with open('output.txt', 'w') as out:
        out.write(str('######## Quick Machine Learning ########\n'))
        out.write(str('----------- Summray Report ------------\n'))
        out.write(var.stdout)
        out.close()


# ## Deployment

# In[16]:


import pickle


# In[17]:


app_py = ['import numpy as np\n',
 'from flask import Flask, request, jsonify, render_template\n',
 'import pickle\n',
 '\n',
 'app = Flask(__name__)\n',
 "model = pickle.load(open('model.pkl', 'rb'))\n",
 '\n',
 "@app.route('/')\n",
 'def home():\n',
 "    return render_template('index.html')\n",
 '\n',
 "@app.route('/predict',methods=['POST'])\n",
 'def predict():\n',
 "    '''\n",
 '    For rendering results on HTML GUI\n',
 "    '''\n",
 '    int_features = [int(x) for x in request.form.values()]\n',
 '    final_features = [np.array(int_features)]\n',
 '    prediction = model.predict(final_features)\n',
 '\n',
 '    output = round(prediction[0], 2)\n',
 '\n',
 "    return render_template('index.html', prediction_text='Prediction:  {}'.format(output))\n",
 '\n',
 "@app.route('/predict_api',methods=['POST'])\n",
 'def predict_api():\n',
 "    '''\n",
 '    For direct API calls trought request\n',
 "    '''\n",
 '    data = request.get_json(force=True)\n',
 '    prediction = model.predict([np.array(list(data.values()))])\n',
 '\n',
 '    output = prediction[0]\n',
 '    return jsonify(output)\n',
 '\n',
 'if __name__ == "__main__":\n',
 '    app.run(debug=True)']


# In[18]:


def index_file(title):
    htm = ['<!doctype html>\n',
     '<html lang="en">\n',
     '<head>\n',
     '    <meta charset="UTF-8">\n',
     '    <title>Ml Deploy</title>\n',
     '    <style>\n',
     '        body{\n',
     '            margin: 0;\n',
     '            padding: 0;\n',
     '            }\n',
     '        body:before{\n',
     "            content: '';\n",
     '            position: fixed;\n',
     '            width: 100vw;\n',
     '            height: 100vh;\n',
     '            background: -webkit-linear-gradient(rgb(92, 57, 219),rgb(173, 218, 210),rgb(235, 193, 56)); \n',
     '            background: -o-linear-gradient(rgb(92, 57, 219),rgb(173, 218, 210),rgb(235, 193, 56)); \n',
     '            background: -moz-linear-gradient(rgb(92, 57, 219),rgb(173, 218, 210),rgb(235, 193, 56)); \n',
     '            background: linear-gradient(rgb(92, 57, 219), rgb(173, 218, 210),rgb(235, 193, 56)); \n',
     '            background-color:rgb(46, 236, 157); \n',
     '            background-position: center center;\n',
     '            background-repeat: no-repeat;\n',
     '            background-attachment: fixed;\n',
     '            background-size: relative;\n',
     '            -webkit-filter: blur(10px);\n',
     '            -moz-filter: blur(10px);\n',
     '            -o-filter: blur(10px);\n',
     '            -ms-filter: blur(10px);\n',
     '            filter: blur(10px);\n',
     '        }\n',
     '        .form\n',
     '        {\n',
     '            position: absolute;\n',
     '            top:5%;\n',
     '            left:30%;\n',
     '            margin-bottom: 20px;\n',
     '            align-self :center;\n',
     '            /*transform: translate(-50%,-50%);*/\n',
     '            width: 400px;\n',
     '            height: relative;\n',
     '            padding: 50px 40px;\n',
     '            box-sizing: border-box;\n',
     '            background: rgba(0,0,0,.55);\n',
     '        \n',
     '        }\n',
     '        .form h2 {\n',
     '            margin: 0;\n',
     '            padding: 0 0 20px;\n',
     '            color:white;\n',
     "            font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;\n",
     '            text-align: center;\n',
     '            text-transform: uppercase;\n',
     '        }\n',
     '        .form p\n',
     '        {\n',
     '            margin: 0;\n',
     '            padding: 0;\n',
     '            font-weight: bold;\n',
     '            color: #fff;\n',
     '        }\n',
     '        .form input\n',
     '        {\n',
     '            width: 100%;\n',
     '            margin-bottom: 20px;\n',
     '        }\n',
     '        .form input[type="text"]\n',
     '        {\n',
     '            border: none;\n',
     '            border-bottom: 1px solid #fff;\n',
     '            background: transparent;\n',
     '            outline: none;\n',
     '            height: 40px;\n',
     '            color: #fff;\n',
     '            font-size: 16px;\n',
     '        }\n',
     '        .form input[type="submit"] {\n',
     '            height: 40px;\n',
     '            color: whitesmoke;\n',
     '            font-weight: bold;\n',
     '            font-size: 15px;\n',
     '            background: rgb(28, 235, 131);\n',
     '            cursor: pointer;\n',
     '            border-radius: 25px;\n',
     '            border: none;\n',
     '            outline: none;\n',
     '            margin-top: 5%;\n',
     '        }\n',
     '        .form a\n',
     '        {\n',
     '            color: #fff;\n',
     '            font-size: 14px;\n',
     '            font-weight: bold;\n',
     '            text-decoration: none;\n',
     '        }\n',
     '        input[type="checkbox"] {\n',
     '            width: 20%;\n',
     '        }\n',
     '        h2{\n',
     '            color:white;\n',
     '            font-family: sans-serif;\n',
     '        }\n',
     '    </style>\n',
     '</head>\n',
     '<body>\n',
     '    <div class="form">\n',
     '        <h2>'+title+'</h2>\n',
     '        <form action="{{ url_for(\'predict\')}}" method="post">\n',
     '            \n',
     '            <input type="submit" name="" value="Predict">\n',
     '        </form>\n',
     '        <form>\n',
     '            <p>{{ prediction_text }}</p>\n',
     '        </form>\n',
     '    </div>\n',
     '</body>\n',
     '</html>']
    return htm


# In[19]:


def deploy_one(model,X,title=None):
    '''
    Deploy one Method, Deploys Machine Learning Models to Web App using Flask
    All files will be stored in deployment-files folder created automatically in notebook running directory
    usage:
        deploy_form(model,X,'title')
    model -> Machine Learning Model
    X     -> Features (Note Features should be iterable)
                eg - list,dataframe of Features(X_train)
  'title' -> Title For Html Deplyment UI (optional)
              Default is 'Make Predictions'
    '''
    try:
        if title != None:
            htm_template = index_file(title)
        else:
            htm_template = index_file('Make Predictions')
        
        os.mkdir('deployment-files')
        os.mkdir('deployment-files/templates')
        pickle.dump(model, open('deployment-files/model.pkl','wb'))
        with open('deployment-files/app.py','w') as mf:
            for k in app_py:
                mf.write(k)
            mf.close()
        with open("deployment-files/templates/index.html", "w") as f1:
            for i in htm_template:
                f1.write(i)
                for k in i.split(' '):
                    if k == '<form':
                        for i in X:
                            htmline = '<input type="text" name="'+str(i)+'" placeholder="'+str(i)+'" required="required" autocomplete="off" />' 
                            f1.write(str(htmline)+'\n')
            f1.close()
        print('All Deployment Files Created Successfully\n')
        print('Files Created and stored in : /deployment-files named Folder\n')
        print('Open deployment-files folder and open command prompt and run "python app.py" copy link and paste in browser to run app, enjoy.')
        print('\nFiles created:\n 1. model.pkl - Model file using pickle\n 2.app.py - Flask server file\n3.index.html - User Interface file.')
    except:
        print('There was an error Generating Deployment Files')
        print('Note:\n Delete "Deployment-files" folder if already exists.')


# In[20]:


### just in case no plots displayed
import plotly.offline as pyo
pyo.init_notebook_mode()

