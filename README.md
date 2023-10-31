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
Remember to alter the 
## File structure of the project
1) db_utils.py
2) Column_transform.py (accompanied by EDA_dtype_transformed.pkl)
## License information
