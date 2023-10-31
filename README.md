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

In addition to DataTransform, the project also includes a DataFrameInfo class for obtaining useful information from the DataFrame and a Plotter class for creating various plots for data visualization. These outputs infromed the decisions onm how to deal with null values via imputation or dropping columns/rows.

## Installation instructions
To install and run this project, you will need Python 3 and the following Python libraries installed:

pandas
SQLAlchemy
PyYAML
csv
dateutil
pickle
## Usage instructions
Clone this repository to your local machine.
Use the .py scripts in the order they are listed.
Remember to alter the file paths for the location of your own directory and relevant files
## File structure of the project
1. db_utils.py: This script manages the creation of the engine object for accessing an Amazon RDS database. New dataframe is saved to loan_payments.csv.
2. Column_transform.py: This script reads in the data from a CSV file, performs various transformations using the DataTransform class, and saves the transformed DataFrame to a pickle file (EDA_dtype_transformed.pkl).
3. Data_Frame_Info.py This script contains the Plotter class and removes null values from the dataset via dropping columns or rows and imputing null values. The updated dataframe is saved to EDA_nulls_removed.pkl
## License information
