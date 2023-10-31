import pandas as pd
import csv
from dateutil import parser
import pickle

payments = pd.read_csv(r"C:\Users\Dr Dankali\AICORE\EDA_Finance\exploratory-data-analysis---customer-loans-in-finance\loan_payments.csv")
payments
new_df = payments
# Here the DataTransform class allows for simple conversion of column data types adn adjustments to their
# accuracy
class DataTransform():
    def __init__(self, df):
        self.df = df

    def to_float64(self, colname):
        self.colname = colname
        self.df[self.colname] = self.df[self.colname].astype("float64")
        print(self.df[self.colname].dtypes)
        print(f"{self.colname} sample value: {self.df[self.colname].iloc[0]}")

    
    def round_to_2(self, colname):
        self.colname = colname
        self.df[self.colname] = self.df[self.colname].round(2)
        print(self.df[self.colname].dtypes)
        print(f"{self.colname} sample value: {self.df[self.colname].iloc[0]}")
    
    def round_to_none(self, colname):
        self.colname = colname
        self.df[self.colname] = self.df[self.colname].round(0)
        print(self.df[self.colname].dtypes)
        print(f"{self.colname} sample value: {self.df[self.colname].iloc[0]}")
        

    def to_int64(self, colname):
        self.colname = colname
        self.df[self.colname] = self.df[self.colname].astype("int64")
        print(self.df[self.colname].dtypes)
        print(f"{self.colname} sample value: {self.df[self.colname].iloc[0]}")
    
    def percentage(self, colname):
        self.colname = colname
        self.df[self.colname] = self.df[self.colname].map('{:2%}'.format)
        print(self.df[self.colname].dtypes)
        print(f"{self.colname} sample value: {self.df[self.colname].iloc[0]}")
    
    def to_category(self, colname):
        self.colname = colname
        self.df[self.colname] = self.df[self.colname].astype("category")
        print(self.df[self.colname].dtypes)
        print(f"{self.colname} sample value: {self.df[self.colname].iloc[0]}")
    
    def parse_and_format_date(self, date_string):
        if isinstance(date_string, str):
            try:
                parsed_date = parser.parse(date_string)
                return parsed_date.strftime('%Y-%m')
            except ValueError:
                return None
        else:
            try:
                parsed_date = parser.parse(str(date_string))
                return parsed_date.strftime('%Y-%m')
            except ValueError:
                return None

    def to_YYYY_MM(self, colname):
        self.df[colname] = self.df[colname].apply(self.parse_and_format_date)
        print(self.df[self.colname].dtypes)
        print(f"{self.colname} sample value: {self.df[self.colname].iloc[0]}")

# Create an instance of the DataTransform class
new_df1 = DataTransform(new_df)
# Converting columns to "float64"
new_df1.to_float64("loan_amount")
new_df1.to_float64("annual_inc")

# Rounding columns to a reasonable 2 decimal places (for currency)
cols_round_to_2 = ["loan_amount", "annual_inc", "funded_amount", "funded_amount_inv", "instalment", "total_rec_late_fee", "recoveries", "collection_recovery_fee"]

def mass_round(object, column_list):
    for col in column_list:
        object.round_to_2(col)
mass_round(new_df1, cols_round_to_2)

# Converting columns to category based on unique values
cols_to_category = ["grade", "sub_grade", "employment_length", "home_ownership", "verification_status", "loan_status", "purpose", "policy_code", "application_type", "mths_since_last_major_derog"]

def mass_category(object, column_list):
    for col in column_list:
        object.to_category(col)
mass_category(new_df1, cols_to_category)


# Parsing and converting columns to ISO YYYY-MM format
cols_to_YYYY_MM = ["issue_date", "earliest_credit_line", "last_payment_date", "next_payment_date", "last_credit_pull_date"]
def mass_to_YYYY_MM(object, column_list):
    for col in column_list:
        object.to_YYYY_MM(col)
mass_category(new_df1, cols_to_YYYY_MM)


# Here will make a final adjustment and correct the name of the "instalment" column to "installment"
new_df.rename(columns={"instalment": "installment"}, inplace=True)

# Saving new dataframe to .pkl format
with open(r"C:\Users\Dr Dankali\AICORE\EDA_Finance\exploratory-data-analysis---customer-loans-in-finance\EDA_dtype_transformed.pkl", 'wb') as file:
    pickle.dump(new_df, file)

