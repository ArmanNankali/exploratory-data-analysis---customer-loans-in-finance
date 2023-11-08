import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from Data_Frame_Info import Plotter
import json
from Data_Frame_Info import DataFrameInfo
from sklearn.preprocessing import PowerTransformer

with open(r"C:\Users\Dr Dankali\AICORE\EDA_Finance\exploratory-data-analysis---customer-loans-in-finance\EDA_nulls_removed.pkl", 'rb') as file:
    nulls_removed_loans = pickle.load(file)

#--------------------------------------------SKEW-------------------------------------------
# Now we have to deal with the skewness of the data
# The following function will apply a skew test to all variables in the dataframe
nrl1 = Plotter(nulls_removed_loans)
def total_skew(df, object):
    for col in df:
        try:
            object.skew_plot(col)
        except:
            pass
total_skew(nulls_removed_loans, nrl1)

# The variables have a wide range of skewness from 0.19 all the way up to 27.24
# Using the following guideline, the variables were grouped into:
# -1.5 to 1.5 low skew
# -5 to 5 skewed
# below -5 or above 5 was very skewed

# Here we introduce the DtTransform class which contain three different transformations for skewed data
# Each of Box-Cox, Yeo-Johnson and Log transformations have two versions, one of which returns the value of the 
# skew after the transformation. This will inform our descisions on which transformation to use to 
# permanently transform the column. The accompanying column converion method will then be chosen
# to transform the column in the dataframe
# There are two dictionaries and a list in the initialiser, these will be explained in more detail later.
class DtTransform():
    def __init__(self, df):
        self.df = df
        self.Log_list = []
        self.YJH_transformers = {}
        self.BXC_transformers = {}

    def log_transform_check(self, col):
        self.col = col
        log_population = self.df[self.col].map(lambda i: np.log(i) if i > 0 else 0)
        t=sns.histplot(log_population,label="Log transformed Skewness: %.2f"%(log_population.skew()) )
        t.legend()
    
    def col_to_log_transform(self, col):
        self.col = col
        self.df[self.col] = self.df[self.col].map(lambda i: np.log(i) if i > 0 else 0)
        self.Log_list.append(self.col)
        t=sns.histplot(self.df[self.col],label="Log transformed Skewness: %.2f"%(self.df[self.col].skew()) )
        t.legend()
    
    def yeo_johnson_transform_check(self, col):
        self.col = col
        pt = PowerTransformer(method="yeo-johnson")
        yeojohnson_population = self.df[self.col].values.reshape(-1, 1)
        yeojohnson_population = pt.fit_transform(yeojohnson_population)
        t=sns.histplot(yeojohnson_population,label=f"{self.col} Yeojohnson Population Skewness: %.2f"%(self.df[self.col].skew()) )
        t.legend()

    def col_to_yeo_johnson_transform(self, col):
        self.col = col
        pt = PowerTransformer(method="yeo-johnson")
        data = self.df[self.col].values.reshape(-1, 1)
        self.df[self.col] = pt.fit_transform(data)
        self.YJH_transformers[col] = pt
        t=sns.histplot(self.df[self.col],label=f"{self.col} Yeojohnson Population Skewness: %.2f"%(self.df[self.col].skew()) )
        t.legend()
    

    def box_cox_transform_check(self, col):
        self.col = col
        pt = PowerTransformer(method="box-cox")
        boxcox_population = self.df[self.col].values.reshape(-1, 1)
        boxcox_population = pt.fit_transform(boxcox_population)
        t=sns.histplot(boxcox_population,label=f"{self.col} Box-Cox Skewness: %.2f"%(self.df[self.col].skew()) )
        t.legend()

    def col_to_box_cox_transform(self, col):
        self.col = col
        pt = PowerTransformer(method="box-cox")
        data = self.df[self.col].values.reshape(-1, 1)
        self.df[self.col] = pt.fit_transform(data)
        self.BXC_transformers[col] = pt
        t=sns.histplot(self.df[self.col],label=f"{self.col} Box-Cox Skewness: %.2f"%(self.df[self.col].skew()) )
        t.legend()

    def z_score(self, col):
        self.col = col
        mean_y = np.mean(self.col)
        std_y = np.std(self.col)
        z_scores = [(yi - mean_y) / std_y for yi in self.col]
        print(z_scores)

# With the code below we instantiate the class but also quickly obtain the skewness of all the 
# possible transformations to let us choose the appropriate one.
# It could be argued that a function which automatically selects the lowest skewness possible 
# would be less cumbersome than checking each column seperately. However, there is more nuance to 
# the transformation process than "lowest skew possible". When a yeo-johnson transformation provides 
# a greater reduction of skew than a regular log transformation by 0.01 it would be favourable to 
# select the log transformation as it is simpler to interperet the results and logic.

nrl2 = DtTransform(nulls_removed_loans)
#x = "column name"
#nrl2.log_transform_check(x)
#nrl2.yeo_johnson_transform_check(x)
#nrl2.box_cox_transform_check(x)

# Using this function, the following list was made with each variable and the chosen transformation
# method. All variables are grouped based on which of the three transformations will be used.
# A function has been created to transform each column within a group all at once.

#----------------------------------------L.O.G.-----------------------------------------------
#Skew of out_prncp column is 2.352525970614482
#YEO johnson reduced skew to 0.53 or log to 0.57
#Skew of out_prncp_inv column is 2.3529464346145614
#YEO johnson reduced skew to 0.53 or log to 0.57
#Skew of annual_inc column is 8.73573692583551
#log transformed down to 0.14

log_list = ["out_prncp", "out_prncp_inv", "annual_inc"]
def mass_log_transform(class_object, col_list):
    for col in col_list:
        class_object.col_to_log_transform(col)

mass_log_transform(nrl2, log_list)

#------------------------------------B.O.X.-C.O.X--------------------------------------------
#Skew of installment column is 0.9961503404396858
#boxcox reduced skew to -0.02
#Skew of loan_amount column is 0.8043121863503576
#boxcox reduced skew to -0.04
#Skew of funded_amount column is 0.8296639814360702
#boxcox reduced skew to -0.08
#Skew of total_accounts column is 0.7800832932375489
#boxcox reduced skew to -0.01

#Skew of int_rate column is 0.43254487784332657
#boxcox reduced skew to -0.01
#Skew of total_payment column is 1.2692662197762663
#boxcox reduced skew to -0.01

box_cox = ["installment", "loan_amount",  "funded_amount", "total_accounts", "int_rate", "total_payment"]
def mass_box_cox(class_object, col_list):
    for col in col_list:
        class_object.col_to_box_cox_transform(col)

mass_box_cox(nrl2, box_cox)

#-------------------------------Y.E.O-J.O.H.N.S.O.N.------------------------------------------
#Skew of collection_recovery_fee column is 27.794719840726795
#yeo-johnson reduced skew to 3.62
#Skew of collections_12_mths_ex_med column is 20.29806369008403
#yeo-johnson reduced skew to 16.03
#Skew of total_rec_late_fee column is 13.258415394509258
#yeo-johnson reduced skew to 5.29
#Skew of recoveries column is 14.382675354865084
#yeo-johnson reduced skew to 3.48
#Skew of delinq_2yrs column is 5.318817917464191
#yeo-johnson reduced skew to 1.87
#Skew of inq_last_6mths column is 3.243836167526465
#yeo-johnson reduced skew to 0.25
#Skew of total_rec_int column is 2.203625284669197
#yeo-johnson reduced skew to 0.00
#Skew of last_payment_amount column is 2.496078600255455
#yeo-johnson reduced skew to 0.00
#Skew of total_payment_inv column is 1.2584742300393752
#yeo-johnson reduced skew to 0.01
#Skew of total_rec_prncp column is 1.262007269838639
#yeo-johnson reduced skew to -0.02
#Skew of open_accounts column is 1.0594931166880517
#yeo-johnson reduced skew to 0.00
#Skew of funded_amount_inv column is 0.8142509300370537
#yeo-johnson reduced skew to -0.04
#Skew of dti column is 0.18901977139121579
#yeo-johnson reduced skew to -0.09


yeo_johnson_list = ["collections_12_mths_ex_med", "recoveries", "delinq_2yrs", "inq_last_6mths", "total_rec_int", "last_payment_amount", "total_payment_inv", "total_rec_prncp", "open_accounts", "funded_amount_inv", "dti"]
def mass_yeo_johnson(class_object, col_list):
    for col in col_list:
        class_object.col_to_yeo_johnson_transform(col)

mass_yeo_johnson(nrl2, yeo_johnson_list)

# Now, repeating the earlier skew check of the entire dataframe reveals much lower skews
nrl1 = Plotter(nulls_removed_loans)
total_skew(nulls_removed_loans, nrl1)
 
# Each transformation is necessary to reduce skew as much as possible. This becomes very important
# for downstream machine learning pipelines. However, for analysis of raw values (such as money)
# transformed data may be inappropriate. To accomdate for this, each logged variable has been
# added to a list, so they can be easily identified later for application of an exponential transformation
# back to their original values.
# As for Box-Cox and Yeo-Johnson, the power transformer used to transform each column has been stored in a
# dictionary for later inverse-transformation if necessary. 

# First the new dataframe will be saved to a new .pkl file

with open(r"C:\Users\Dr Dankali\AICORE\EDA_Finance\exploratory-data-analysis---customer-loans-in-finance\EDA_skew_removed.pkl", 'wb') as file:
    pickle.dump(nulls_removed_loans, file)

# Now the dictionaries and lists will be saved

# Save to pickle
with open(r"C:\Users\Dr Dankali\AICORE\EDA_Finance\exploratory-data-analysis---customer-loans-in-finance\YJH_transformers.pkl", 'wb') as file:
    pickle.dump(nrl2.YJH_transformers, file)

with open(r"C:\Users\Dr Dankali\AICORE\EDA_Finance\exploratory-data-analysis---customer-loans-in-finance\BXC_transformers.pkl", 'wb') as file:
    pickle.dump(nrl2.BXC_transformers, file)

# Save to json
with open(r"C:\Users\Dr Dankali\AICORE\EDA_Finance\exploratory-data-analysis---customer-loans-in-finance\Log_list.json", 'w') as file:
    json.dump(nrl2.Log_list, file)
