[![GitHub license](https://img.shields.io/github/license/Chintan99/eazeml?label=EazeML)](https://github.com/Chintan99/eazeml/blob/master/LICENSE.txt)
# Eazeml ¯\\\_(ツ)_/¯

eazeml is a Python 3.x based Machine Learning Open Source Library which makes the process of machine learning and Data science Faster, Easy and reduces the difficulty of manual coding.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install eazeml.

```bash
pip install eazeml
```

## Usage

```python
import eazeml as ez

ez.list_func() ## lists all the available Functions
help(eazml) ## Prints all function and usage
```

## Note: 
#### Only Few Functions example are shown below, for all methods example and usage please see this example notebook.

## Some Automated functions
### quick_ml() 
quick_ml method provides automation to data cleaningn & machine learning process here we just need to provide
1. Pandas Dataframe containg data.
2. Target-columnname we want to make a prediction of.
3. flag: which specify prediction type (like 'r' for regression and 'c' for classification).

and thats it, function will automatically clean, process data and give you output as a complete summary of processing done and returns report table and model used for prediction

```python
ez.quick_ml(dataframe,'target_column','flag')
or
report,model = ez.quick_ml(dataframe,'target_column','flag')
```
### quick_pred()
quick_pred method takes cleaned data and prediction flag('r','c') and does prediction using several different algorithms and gives output score for each algorithm in tabular format.
- regression
<img src="https://github.com/Chintan99/eazeml/blob/master/mdimages/report-r.PNG" alt="drawing2" width="600" height='200' />

- classification
<img src="https://github.com/Chintan99/eazeml/blob/master/mdimages/report-c.PNG" alt="drawing1" width="600" height='200' />

```python
X = features
y = target column
## flag('r','c')
ez.quick_pred(X,y,'flag')
```
### kaggle()
kaggle method takes input argument: traindataset, testdataset,target columname and flag('r','c') and gives out as complete summary of steps performed in prediction and data cleaning process and returns predicted values of test dataframe 

```python
ez.kaggle(train,test,'targetcolumn','flag')
or
test_pred,report = ez.kaggle(train,test,'targetcolumn','flag')
## test_pred- consist test file prediction
## report - tabular report score
```
### clean_data()
clean_data automatically cleans data by:
1. removing columns having null value greated than 20%
2. impute median in missing numeric columns
3. impute mode in missing categorical column
```python
dataframe = ez.clean_data(dataframe)
## returns cleaned dataframe
```

### nlp_text()
nlp_text converts Textual data of Column to Tfidf vector after operations:
1. Removing Punctutions
2. Removing Special Symbol/character
3. Lemmatization
4. Coverts Uppercase to lowercase

- only column passed is converted to vector not whole dataframe
- need to manually join output data vector with original dataframe
```python
vec = ez.nlp_text(dataframe,'columname')
## returns cleaned Tfidf vector
```

## Some example of Basic functions
### info() 
info method gives complete information about the data in tabluar format including:
1. Number of Rows,Column
2. Number of Categorical, Numerical Feature
3. null, unique values in each column
4. total missing value and missing value percentage in each column.
5. Plots graph of Missing data in percentage format.

```python
ez.info(dataframe)
```
### getcolumnstype()
getcolumnstype returns the type of columns and store it in list so reserchers not have to waste time in writing code and manully determining columns type.
```python
num,cat = getcolumnstype(dataframe)
## num - list of numeric columns
## cat - list of categorical columns
```
### drop_columns() & dropcolumns()
drop_columns takes argument as Percentage in no and columns name 
1. no - number of null value percentage, above which all columns will be dropped and 
2. (optional) columns name - specify columns sepreated with ',' will be dropped
```python
dataframe = ez.drop_columns(dataframe,20,'columnnames')
``` 
- This Example will drop all columns having null value percentage above 20 and drop columns specifiied if don't wan't to use n feature just mention 0 in place of n
```python
df = ez.drop_columns(df,n,'id','name')
```
#### dropcolumns():
it just drop specified columns. example:
```python
df = ez.drop_columns(df,'id','name','tp','dsd')
```
### fillnulls()
fillnulls method performs imputation operation of mean ,median,mode or specified value on specified columns together and returns imputed dataframe:
- Example 1: filling median in columns
```python
df = fillnulls(df,'median','col1','col2',...)
# fill median of respective column to respective column
```
- Example 2: filling mean in columns
```python
df = fillnulls(df,'mean','col1','col2',...)
# fill mean of respective column to respective column
```
- Example 3: filling mode in columns
```python
df = fillnulls(df,'mode','col1','col2',...)
# fill mode of respective column to respective column
```
- Example 4: filling specified value in columns
```python
df = fillnulls(df,'unknown','col1','col2',...)
# unknown will be filled inplace of null values in col2, col1
```
### extract_number()
Method Extract number from column removing any extra string,special char,symbol etc and return cleaned column
```python
df = ez.extract_number(df,'income')
# removes $symbol and return cleaned column dataframe
```
### get_IQR
Prints IQR (Inter Qurantile Range) values table for each column of dataframe 
```python
df = ez.get_IQR(dataframe)
```
### show_outlier & remove_outlier
- show_outlier()

Method Prints number of outlier in each column of dataframe
```python
ez.show_outlier(dataframe)
```
- remove_outlier()

Method removes all outliers from dataframe and returns Normally Distributed Dataframe and prints IQR range table.
```python
newdf = ez.remove_outlier(dataframe)
# stores outlier free dataframe in newdf
```
### stat_models()
Prints Statastical Significance table(OLS table) for each column of dataframe with respect to target column 
```python
X = features
y = target column
ez.stats_models(y,X)
```
### VIF()
Method Prints VIF(Variation Inflation Variance)value of each column in Table format for each column of dataframe
```python
ez.VIF(dataframe)
```
### corr_heatmap()
Plots Corelation Heatmap of given dataframe.

arguments:

1. Dataframe
2. Style: 
    - 'interactive': uses plotly and plots interactive heatmap.
    - 'basic': prints colorized tabular format heatmap.
    - none: uses seaborn to plot annoted heatmap.

```python
ez.corr_heatmap(df)
# Plots heatmap with seaboron
ez.corr_heatmap(df,'interactive')
# plots interactive heatmap with plotly
ez.corr_heatmap(df,'basic')
# prints colorized tabluar correaltion heatmap
```
### confusion_mat()
Method Plots Visualized Confusion Matrix.

example:
![binary](https://github.com/Chintan99/eazeml/blob/master/mdimages/cm2.PNG =100x20)
![multiclass](https://github.com/Chintan99/eazeml/blob/master/mdimages/cm1.PNG =100x20)
```python
#y_true- Actual Value
#y_pred- Predicted Value
ez.confusion_mat(y_true,y_pred)
# you can change color using cmap
```
### roc_curve_graph
Method Plots AUC-ROC plot for y_true,y_pred
![auc plot](https://github.com/Chintan99/eazeml/blob/master/mdimages/auc.PNG)
- only works with Binary Classification
```python
ez.roc_curve_graph(y_true,y_pred)
```
### box_hist_plot()
method plots interactive Box plot and histogram
using plotly library
```python
ez.box_hist_plot(dataframe,'col1','col2',...)
# plots box plot and histogram for all specified column of dataframe.
```

### classification() & regression()
- classification():

    It is menu based classification, asks user to which model to use for prediction.

    It returns y_predicted,y_actual, model used for classification

    It also prints all classification score metrics and confusion matrix by default 
```python
# X = features
# y = target column
y_pred,y_actual,model = ez.classification(X,y)
```
- regression():

    It is menu based regression method, asks user for which model to use for prediction.

    It retruns y_predicted,y_actual, model used for prediction

    It also prints all regression score metrics by default
```python
# X = features
# y = target column
y_pred,y_actual,model = ez.regresssion(X,y)
```
### RFE()
Methods Perfroms Recurrsive Feature Elimination and returns best n columns specified
```python
rfecolumns = ez.rfe(model,X_train,y_train,n)
# n- number of features to be selected (defualt n=10)
# model - model to use RFE with.
```
### classification_result() & regression_result()
Prints all scoring Metrics for when passed y_true & y_predicted
- classification_result()
    
    Evaluation Matrics:
    
    - F1 score
    - AUC-ROC score
    - classification report
    - plots confusion matrix
    - plots auc graph (only if binary classification)
```python
ez.classification_result(y_true,y_pred)
```
- regression_result()
    
    Evaluation Matrics:
    
    - R2 score
    - Mean Squared Error score
    - Root Mean Squared Error score
    - Root Mean Squared log Error score
```python
ez.regression_result(y_true,y_pred)
```
### Machine Learning ALgorithms
Eazeml provides use of each algoritm with custom parameter
usage:
```python
# first need to define these variables
X = Feature
y = Targetcolumn
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=10)
ez.regression_result(y_true,y_pred)
```
- These Machine Learning Algorithms are already set to best param grid from experienced users.
- 'reg' defines regression & 'classifier' defines classifier algorithm.
```python
## pass X_train,y_train,X_test,y_test in which ever algorithm you need.
y_pred,model = ez.linear_reg(X_train,y_train,X_test,y_test)
## return y_predicted and model for further usage.
ez.logistic_reg(X_train,y_train,X_test,y_test)
ez.randomforest_classifier(X_train,y_train,X_test,y_test)
ez.randomforest_reg(X_train,y_train,X_test,y_test)
ez.xgb_classifier(X_train,y_train,X_test,y_test)
ez.xgb_reg(X_train,y_train,X_test,y_test)
ez.lgb_classifier(X_train,y_train,X_test,y_test)
ez.lgb_reg(X_train,y_train,X_test,y_test)
ez.Gradient_boost_classifier(X_train,y_train,X_test,y_test)
ez.Gradient_boost_reg(X_train,y_train,X_test,y_test)
ez.adaboost_classifier(X_train,y_train,X_test,y_test)
ez.adaboost_reg(X_train,y_train,X_test,y_test)
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
