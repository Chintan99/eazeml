# Eazeml ¯\\\_(ツ)_/¯

eazeml is a Python 3.x based Machine Learning Open Source Library which makes the process of machine learning and Data science easy and reduces the hassle of coding for new commers

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
#### Only Few Functions example are shown below, for all methods example and usage please so this example notebook.

## Some Automated functions
#### quick_ml() 
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
#### quick_pred()
quick_pred method takes cleaned data and prediction flag('r','c') and does prediction using several different algorithms and gives output score for each algorithm in tabular format.

```python
ez.quick_pred(features(X),target(y),'flag')
```
#### kaggle()
kaggle method takes input of traindataset, testdataset,target columname and flag('r','c') and gives out as complete summary of steps performed and returns predicted values of test dataframe 

```python
ez.kaggle(train,test,'targetcolumn','flag')
or
test_pred,report = ez.kaggle(train,test,'targetcolumn','flag')
```
#### clean_data
#### nlp_text

## Some example of Basic functions
#### info() 
info method gives complete information about the data in tabluar format including:
1. Number of Rows,Column
2. Number of Categorical, Numerical Feature
3. null, unique values in each column
4. total missing value and missing value percentage in each column.
5. Plots graph of Missing data in percentage format.

```python
ez.info(dataframe)
```
#### getcolumnstype
#### drop_columns() & dropcolumns()
#### fillnulls
#### extract_number
#### get_IQR
#### show_outlier & remove_outlier
#### stat_models
#### VIF
#### corr_heatmap
#### confusion_mat
#### roc_curve_graph
#### classification() & regression()
#### classification_result & regression_result

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
