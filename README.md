# exploratory-data-analysis---customer-loans-in-finance
armannankali/exploratory-data-analysis---customer-loans-in-finance

## Table of Contents, if the README file is long
Table of Contents
Project Description
Installation Instructions
Usage Instructions
File Structure
License Information
## A description of the project: what it does, the aim of the project, and what you learned
This project aims to perform an exploratory data analysis (EDA) on customer loans data in the finance sector. The main goal is to uncover patterns, relationships, and anomalies in the loan data to aid in decision-making processes related to loan approvals, pricing, and risk management.

The first task involves accessing an Amazon RDS database using provided credentials and downloading the database locally into a pandas DataFrame. A class is used to manage the creation of the engine object, simplifying future access to any RDS database with minimal modifications.

In order to perform various transformations on the data a custom DataTransform class is used. This class provides methods for converting column data types, rounding numerical values to a specified number of decimal places, converting columns to categories based on unique values, and parsing and converting date columns to a specific format.

The project also includes a DataFrameInfo class for obtaining useful information from the DataFrame and a Plotter class for creating various plots for data visualization. These outputs infromed the decisions onm how to deal with null values via imputation or dropping columns/rows.

The latest addition is dataframe_skew.py, which handles skewness in the data. It includes functions for log transformation, Box-Cox transformation, and Yeo-Johnson transformation. The transformed dataframe and the lambda values used for the transformations are saved for easy reversal of transformations later if needed.

Dataframe_outliers.py handles outliers in the data. It includes functions for removing outliers based on the D’Agostino’s K-squared test and the IQR method. The processed dataframe is saved for easy access later.

Dataframe_colinearity.py explores colinearity by first examining the correlation matrix. From this it was apparent several variables were storngly correlated. By exploring these relationships further via VIF scoring it was discerned that some predictor variables are storngly correlated and could be dropped. However these variables were kept as they are necessary for later analysis.

EDA_deeper_insights.py: Here the data is analysed to: explore current state of the loans, calculate loss, calculate projected loss, calculate possible loss. This file also explores trends in the data to discern factors that might predict a loan becoming a loss to the company. First the columns are inverse-transformed to obtain the original values. Then the data is filtered into subsets of loans that fit into: loss, late/risky, current/full paid. Using correlation matrices to identify any correlated variables and chi squared tests to probe each subset to quanitfy the relationships between loan status and potential predictors. 
## Installation instructions
To install and run this project, you will need Python 3 and the following Python libraries installed:

pandas
SQLAlchemy
PyYAML
csv
dateutil
pickle
pickle
numpy
seaborn
scipy
json
sklearn
## Usage instructions
Clone this repository to your local machine.
Use the .py scripts in the order they are listed.
Remember to alter the file paths for the location of your own directory and relevant files
## File structure of the project
1. db_utils.py: This script manages the creation of the engine object for accessing an Amazon RDS database. New dataframe is saved to loan_payments.csv.
2. Column_transform.py: This script reads in the data from a CSV file, performs various transformations using the DataTransform class, and saves the transformed DataFrame to a pickle file (EDA_dtype_transformed.pkl).
3. Data_Frame_Info.py This script contains the Plotter class and removes null values from the dataset via dropping columns or rows and imputing null values. The updated dataframe is saved to EDA_nulls_removed.pkl
4. Dataframe_skew.py: This script handles skewness in the data by applying log transformation, Box-Cox transformation, or Yeo-Johnson transformation based on skewness of each column in your dataframe. The transformed dataframe and lambda values used for transformations are saved.
5. Dataframe_outliers.py: This script handles outliers in the data by applying D’Agostino’s K-squared test and IQR method based on characteristics of each column in your dataframe. The processed dataframe is saved.
6. Dataframe_colinearity.py: Thi script makes use of the correlation matrix from the Plotter class and further investigates colinearity using VIF to inform which variables are sufficiently correlated where one variable can be dropped.
7. EDA_deeper_insights.py: Here the data is analysed to: explore current state of the loans, calculate loss, calculate projected loss, calculate possible loss and decipher indicators of loss. 
## License information
MIT License
