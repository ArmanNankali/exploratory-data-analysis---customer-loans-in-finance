import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import statsmodels.formula.api as smf
from Data_Frame_Info import Plotter

with open(r"C:\Users\Dr Dankali\AICORE\EDA_Finance\exploratory-data-analysis---customer-loans-in-finance\EDA_skew_removed.pkl", 'rb') as file:
    unskewed_loans = pickle.load(file)


unskewed_loans.set_index('id', inplace=True)

# Having imported the Plotter class we can visualise all the columns with histograms to identify outliers

ul1 = Plotter(unskewed_loans)

unskewed_loans

def all_histograms(object, df):
    for col in df:
        try:
            object.histogram(col)
        except:
            pass

all_histograms(ul1, unskewed_loans)

# Using the D’Agostino’s K-squared test we can check if the data is normally distributed
def all_dak2(object, df):
    dak2_success_list = []
    dak2_error_list = []
    for col in df:
        try:
            object.dak2(col)
            dak2_success_list.append(col)
        except:
            print(f"{col} unable to perfrom dak2")
            dak2_error_list.append(col)
    return dak2_success_list, dak2_error_list

all_dak2(ul1, unskewed_loans)

# The following variables have failed the dak2 test for a normal distribution
# total_payment_inv : Statistics=1.111, p=0.574
# total_rec_int : Statistics=0.093, p=0.954
# last_payment_amount : Statistics=4.790, p=0.091

# Outliers in these variables will be removed using the IQR

# The following code will remove all rows with data points that lie 1.5 interquartile range
# above the 3rd or below the 1st quartile in the appropriate fields.

def IQR_outlier_remover(df, col_list):
    for col in df:
        if col in col_list:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1       
            df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

    return df
col_list1 = ["total_payment_inv", "total_rec_int", "last_payment_amount"]
unskewed_loans = IQR_outlier_remover(unskewed_loans, col_list1)
# %%
unskewed_loans
# %%
# As for the rows which passed the dak2 test, the outliers will be removed if they are above a zscore of 
# 3 or below a score of -3. id, member_id and policy_code have been excluded.
col_list2 = [
  "loan_amount",
  "funded_amount",
  "funded_amount_inv",
  "int_rate",
  "installment",
  "annual_inc",
  "dti",
  "delinq_2yrs",
  "inq_last_6mths",
  "open_accounts",
  "total_accounts",
  "out_prncp",
  "out_prncp_inv",
  "total_payment",
  "total_payment_inv",
  "total_rec_prncp",
  "total_rec_int",
  "total_rec_late_fee",
  "recoveries",
  "collection_recovery_fee",
  "last_payment_amount",
  "collections_12_mths_ex_med"
]


def drop_col_zscore_outlier(object, df, col_list):
    mask = pd.Series([True]*len(df), index=df.index)
    for col in col_list:
        mask &= ~object.z_score(col)

    df = df[mask]
    return df

drop_col_zscore_outlier(ul1, unskewed_loans, col_list2)

# The new dataframe is saved for later analysis of colinearity
with open(r"C:\Users\Dr Dankali\AICORE\EDA_Finance\exploratory-data-analysis---customer-loans-in-finance\EDA_outliers_removed.pkl", 'wb') as file:
    pickle.dump(unskewed_loans, file)