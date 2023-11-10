import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import yeojohnson
from Data_Frame_Info import Plotter
import json
import matplotlib as plt
from Data_Frame_Info import DataFrameInfo
from sklearn.preprocessing import PowerTransformer
from scipy.stats import chi2_contingency
import pandas as pd

with open(r"C:\Users\Dr Dankali\AICORE\EDA_Finance\exploratory-data-analysis---customer-loans-in-finance\EDA_outliers_removed.pkl", 'rb') as file:
    loan_payments = pickle.load(file)
loan_payments

with open(r"C:\Users\Dr Dankali\AICORE\EDA_Finance\exploratory-data-analysis---customer-loans-in-finance\BXC_transformers.pkl", "rb") as file:
  BXC_transformers = pickle.load(file)
BXC_transformers

with open(r"C:\Users\Dr Dankali\AICORE\EDA_Finance\exploratory-data-analysis---customer-loans-in-finance\YJH_transformers.pkl", "rb") as file:
  YJH_transformers = pickle.load(file)
YJH_transformers

with open(r"C:\Users\Dr Dankali\AICORE\EDA_Finance\exploratory-data-analysis---customer-loans-in-finance\Log_list.json", "r") as file:
  Log_list = json.load(file)
Log_list
 
# For all of the variables that have been transformed, a power transformer or list will
# be used to perform the inverse transformation, strictly for the purpose of working with
# original numbers. The following class will enable this

class Inverse_transfrom():
  def __init__(self, df):
    self.df = df

  def inverse_log(self, col):
    self.col = col
    self.df[self.col] = 10 ** self.df[col]

  def inverse_box_cox_transform(self, col, BXC_transformers):
    self.col = col
    self.BXC_transformers = BXC_transformers
    pt = self.BXC_transformers[self.col]
    data = self.df[col].values.reshape(-1, 1)
    self.df[col] = pt.inverse_transform(data)

  def inverse_yeo_johnson_transform(self, col, YJH_transformers):
    self.col = col
    self.YJH_transformers = YJH_transformers
    pt = self.YJH_transformers[self.col]
    data = self.df[col].values.reshape(-1, 1)
    self.df[col] = pt.inverse_transform(data)

loans_analysis = loan_payments.copy()
lp1 = Inverse_transfrom(loans_analysis)


def mass_yjh(df, object):
   for col in df:
      if col in YJH_transformers:
        object.inverse_yeo_johnson_transform(col, YJH_transformers)

mass_yjh(loans_analysis, lp1)

def mass_bxc(df, object):
   for col in df:
      if col in BXC_transformers:
        object.inverse_box_cox_transform(col, BXC_transformers)

mass_bxc(loans_analysis, lp1)

def all_log_inverse(df, object):
   for col in df:
      if col in Log_list:
         object.inverse_log(col)

all_log_inverse(loans_analysis, lp1)

lp_analysis = loans_analysis.copy()
# CURRENT STATE OF THE LOANS--------------------------------------------------------------------------
# Here we have a summary of the percentage that all total_payment combined make up of all funded_amounts combined
total_payment_percentage = round((lp_analysis["total_payment"].sum()/lp_analysis["funded_amount"].sum())*100, 2)
# Here we have a summary of the percentage that all total_payment_inv combined make up of all funded_amount_inv combined
total_payment_percentage_inv = round((lp_analysis["total_payment_inv"].sum()/lp_analysis["funded_amount_inv"].sum())*100,2)

print(f"Total percentage paid against funded amount: {total_payment_percentage}%")
print(f"Total percentage paid against funded amount invested: {total_payment_percentage_inv}%")

#Total percentage paid against funded amount: 95.95%
#Total percentage paid against funded amount invested: 90.73%

# A new column will be created to store the percentage paid towards funded_amount and funded_amount_inv for later 
# referencing and visualisation
lp_analysis["total_percentage_paid"] = round(((lp_analysis["total_payment"]/lp_analysis["funded_amount"])*100),2)
lp_analysis["total_percentage_paid_inv"] = round(((lp_analysis["total_payment_inv"]/lp_analysis["funded_amount_inv"])*100),2)
#-----------------------------------------------------------------------
# Here the results have been filtered to reflect projections of payments from loans that are not fully paid or
# charged off / defaulted
statuses = ['Current', 'Late (31-120 days)', 'Late (16-30 days)', 'In Grace Period']
# Here the dataframe is filtered
filtered_lp_analysis = lp_analysis[lp_analysis['loan_status'].isin(statuses)]


total_payment_percentage = round((filtered_lp_analysis["total_payment"].sum()/filtered_lp_analysis["funded_amount"].sum())*100, 2)
total_payment_percentage_inv = round((filtered_lp_analysis["total_payment_inv"].sum()/filtered_lp_analysis["funded_amount_inv"].sum())*100,2)
print(f"Total percentage paid against funded amount: {total_payment_percentage}%")
print(f"Total percentage paid against funded amount invested: {total_payment_percentage_inv}%")

# Total percentage paid against funded amount: 77.41%
# Total percentage paid against funded amount invested: 73.41%

import matplotlib.pyplot as plt
loan_statuses = lp_analysis['loan_status'].unique()

def histogram_by_loan_status(df, col, loan_statuses):
    for status in loan_statuses:
        subset = df[df['loan_status'] == status]
        plt.figure()
        sns.histplot(subset[col], kde=False)
        plt.title(f'Histogram for {status}')
        plt.show()

histogram_by_loan_status(lp_analysis, "total_percentage_paid", loan_statuses)

# A histogram is used to visualise the total percentage paid towards the total funded amount

# The dataframe is again, filtered based on loans than are still expected to towards the funded amounts
statuses = ['Current', 'Late (31-120 days)', 'Late (16-30 days)', 'In Grace Period']
filtered_lp_analysis = lp_analysis[lp_analysis['loan_status'].isin(statuses)]
# The total amount paid is added to the installment acrross 6 months for each loan and their combined percentage
# is calculated of the funded_amount
total_payment_percentage_6mths = round(((filtered_lp_analysis["total_payment"].sum()+(filtered_lp_analysis["installment"].sum()*6))/filtered_lp_analysis["funded_amount"].sum())*100, 2)

print(f"Total percentage paid against funded amount in 6 months time: {total_payment_percentage_6mths}%")
# Total percentage paid against funded amount in 6 months time: 96.17%
# CALCULATING_LOSS----------------------------------------------------------------

mask = lp_analysis["loan_status"].isin(["Charged Off", "Does not meet the credit policy. Status:Charged Off" ])

percentage_charged_off = round((lp_analysis.loc[mask, "loan_status"].count()/len(lp_analysis["loan_status"]))*100,2)
print(f"Percetnage of loans charged off: {percentage_charged_off}%")
# Percetnage of loans charged off: 10.71%

percentage_paid_towards_charged_off_loans  = round((lp_analysis.loc[mask, "total_payment"].sum()/lp_analysis.loc[mask, "funded_amount"].sum()*100),2)
print(f"Total percentage paid towards charged off loans against funded_amount: {percentage_paid_towards_charged_off_loans}%")
# Total percentage paid towards charged off loans against funded_amount: 52.55%

#CALCULATIN _PROJECTED_LOSS--------------------------------------------------------

# In order to calculate the projected revenue and loss the term column will need to be converted to int64
# where the category of "36 months" would be converted to just 36
lp_analysis["term"] = lp_analysis["term"].str.replace(" months", "").astype(int)

# Expected revenue is calculated by multiplying the funded_amount by the interest rate
lp_analysis["expected_revenue"] = lp_analysis["funded_amount"] * (1 + (lp_analysis["int_rate"] / 100) * lp_analysis["term"])

# Here the mask will select for only charged off loans
mask = lp_analysis["loan_status"].isin(["Charged Off", "Does not meet the credit policy. Status:Charged Off"])

charged_off_loans = lp_analysis[mask]

charged_off_loans["expected_revenue"].sum()

print(lp_analysis.loc[mask, "expected_revenue"].sum())
# expected revenue for the charged off loans is £629400149.18

# The new columns actual_revenue, lost_revenue and percentage_lost_revenue are created
lp_analysis["actual_revenue"] = lp_analysis["total_payment"]

lp_analysis["lost_revenue"] = lp_analysis["expected_revenue"] - lp_analysis["actual_revenue"]

lp_analysis["percentage_lost_revenue"] = (lp_analysis["lost_revenue"] / lp_analysis["expected_revenue"]) * 100

percentage_of_lost_revenue = (lp_analysis.loc[mask,"lost_revenue"].sum() / lp_analysis.loc[mask,"expected_revenue"].sum()) * 100
print(f"Percentage of lost revenue = {round(percentage_of_lost_revenue,2)}%")
# Percentage of lost revenue = 93.92%
total_potential_increase_in_revenue = lp_analysis.loc[mask,"lost_revenue"].sum()
print(f"The total revenue lost in the charged off loans is £{round(total_potential_increase_in_revenue,2)}")
# The total revenue lost in the charged off loans is £591115740.95

# POSSIBLE_LOSS-------------------------------------------------------------------
# Here customers who have failed to  make their payments are selected
mask_possible_loss = lp_analysis["loan_status"].isin(["Late (31-120 days)", "Late (16-30 days)", "In Grace Period"])

(lp_analysis.loc[mask_possible_loss, "loan_status"].count()/lp_analysis[ "loan_status"].count())*100
# They make up 1.78% of all loans by number

total_unpaid = lp_analysis.loc[mask_possible_loss,"funded_amount"].sum() - lp_analysis.loc[mask_possible_loss,"total_payment"].sum()
print(f"The total value unpaid by these loans is £{round(total_unpaid,2)}")
# The total value unpaid by these loans is £3434088.73
projected_loss = ((lp_analysis.loc[mask_possible_loss,"funded_amount"].sum() - lp_analysis.loc[mask_possible_loss,"total_payment"].sum())/lp_analysis["funded_amount"].sum())*100
print(f"The total percentage of the company portfolio the unpaid loans represent is {round(projected_loss)}%")
# The total percentage of the company portfolio the unpaid loans represent is 1%

mask_potential_total_loss = lp_analysis["loan_status"].isin(["Late (31-120 days)", "Late (16-30 days)", "In Grace Period", "Charged Off", "Does not meet the credit policy. Status:Charged Off", "Default"])
# Here the dataframe is filtered to select only loans that have already defaulted or are late
percentage_of_total_revenue = (lp_analysis.loc[mask_potential_total_loss, "total_payment"].sum()/lp_analysis["total_payment"].sum())*100
print(f"The total percentage of the company portfolio the unpaid late and defaulted loans represent is {round(percentage_of_total_revenue,2)}%")
# The total percentage of the company portfolio the late and defaulted loans represent is 7.6%


# INDICATORS OF LOSS--------------------------------------------------------------------------
# A new copy of the dataframe is created with the loan grade converted to a number
# for inclusion in the correlation matrix, with A being the highest grade and 5 being 
# the lowest. These will then be classed as int64
loans_analysis2 = lp_analysis.copy()

grade_mapping = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
loans_analysis2['grade'] = loans_analysis2['grade'].map(grade_mapping)

loans_analysis2["grade"] = loans_analysis2["grade"].astype("int64")
print(loans_analysis2['grade'].dtypes)

#---------------------------------------------------------------------------------------------

# New subsets of the dataframe will be created based on if they are already a loss, late on payments
# or current/ full paid.
mask_failed_loans = loans_analysis2["loan_status"].isin(["Default", "Charged Off", "Does not meet the credit policy. Status:Charged Off"])
failed_loans2 = loans_analysis2.loc[mask_failed_loans].copy()
failed_loans2

fl2 = Plotter(failed_loans2)

fl2.correlation_matrix("failed_loans")
# In failed loans, there appears to be a negative correlation between grade and int_rate

mask_risky_loans = loans_analysis2["loan_status"].isin(["Late (31-120 days)", "Late (16-30 days)", "In Grace Period"])
risky_loans2 = loans_analysis2.loc[mask_risky_loans].copy()
risky_loans2
rl2 = Plotter(risky_loans2)

rl2.correlation_matrix("risky_loans")
# In risky/late loans, there appears to be a negative correlation between grade and int_rate

fpaid_mask = loans_analysis2["loan_status"] == "Fully Paid"
fpaid_loans2 = loans_analysis2.loc[fpaid_mask].copy()
fpl = Plotter(fpaid_loans2)
fpl.correlation_matrix("fully_paid loans")

current_mask = loans_analysis2["loan_status"] == "Current"
current_loans2 = loans_analysis2.loc[current_mask].copy()
current_loans2
crl = Plotter(current_loans2)
crl.correlation_matrix("current loans")
# Throughout all of these matrices a negative correlation between grade and interest rate can be 
# observed

lp2 = Plotter(loans_analysis2)
lp2.correlation_matrix("all_loans")

# Grade appears to be negatively correalted with out_prncp and out_prncp_inv

# finding the counts of each grade for each subset of loans
current_counts = current_loans2["grade"].value_counts()
risky_counts = risky_loans2["grade"].value_counts()
failed_counts = failed_loans2["grade"].value_counts()


plt.figure(figsize=(10, 6))
plt.bar(current_counts.index, current_counts.values, color='blue', alpha=0.5, label='Current Counts')
plt.bar(risky_counts.index, risky_counts.values, color='red', alpha=0.5, label='Risky Counts')
plt.bar(failed_counts.index, failed_counts.values, color='green', alpha=0.5, label='Failed Counts')
plt.legend()
plt.show()

# The following fucntion will allow us to perform a chi^2 test which will allow us
# determine the difference between observed and expected data in our categorical loan status
# variable
def multi_chi_squared(subset_list, col1, col2):
  for subset in subset_list:
    contingency_table_subset = pd.crosstab(subset[col1], subset[col2])
    chi2, p, dof, expected = chi2_contingency(contingency_table_subset)
    print(f"\nSubset: {subset[col1].unique()[0]}")
    print(f"Chi2 value: {chi2}")
    print(f"P-value: {p}")
    print(f"Degrees of freedom: {dof}")

subset_list1 = [failed_loans2,risky_loans2]
multi_chi_squared(subset_list1, "loan_status", "grade")

# failed_loans2 alone had a p value of less than 0.05, making the chi2 value significant and,
# thus, the relationship between loan status and grade significant
# In order to explore this further we will examine the residuals, to show how much the
# observed counts for each grade in each loan status deviate from the expected distribution

def chi_square_residual(subset, col1, col2):
    contingency_table = pd.crosstab(subset[col1], subset[col2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"\nSubset: {subset[col1].unique()[0]}")
    print(f"Chi2 value: {chi2}")
    print(f"P-value: {p}")
    print(f"Degrees of freedom: {dof}")
    residuals = contingency_table - expected
    standardized_residuals = residuals / np.sqrt(expected)
    print(standardized_residuals)

chi_square_residual(failed_loans2, "loan_status", "grade")

# Charged Off: the only residual above 2 or below -2 is grade 7/G with -2.55, which means this grade
# of loan was observed less frequently than expected. This could mean customers with this loan
# grade are unlikely to have their loans charged off and pose the lower risk for that.
# Default: There are no significant residual values 
# Does not meet the credit policy. Status:Charged Off: Grade 1/G loans have a positive residual of 
# 11.4 and have a much higher than expected frequency. Grade E, F and G loans may all be more likely 
# to be charged off for this reason. Conversely, grade A, B and C loans show negative residuals
# and might be less likely to result in charge off for this reason.

# SUB GRADE--------------------------------------------------------------------------------------
# By switching to sub_grade, greater granularity can be achieved
current_counts_percent = current_loans2["sub_grade"].value_counts()/current_loans2["sub_grade"].count()*100
risky_counts_percent = risky_loans2["sub_grade"].value_counts()/risky_loans2["sub_grade"].count()*100
failed_counts_percent = failed_loans2["sub_grade"].value_counts()/failed_loans2["sub_grade"].count()*100
fpaid_loans2_percent = fpaid_loans2["sub_grade"].value_counts()/fpaid_loans2["sub_grade"].count()*100

df_percent = pd.DataFrame({"risky_counts": risky_counts_percent, "failed_counts": failed_counts_percent, "curent_counts": current_counts_percent, "fully paid": fpaid_loans2_percent})
df_percent.plot(kind="bar")

# Show the plot
plt.show()

# This plot reveals that there are fully_paid and current loans make up a greater proportion of
# sub-grade loans between A1 and A5. From D1 to G5 late and failed loans have a higher percentage
# representation of their subsets.

multi_chi_squared(subset_list1, "loan_status", "sub_grade")
# From the chi2 test, only failed_loans 2 has a significant p value, the residuals
# will be investigated further.
chi_square_residual(failed_loans2, "loan_status", "sub_grade")
# Charged Off:  G5 oans are less frequent that expected.
# Default: There are no significant residual values.
# Does not meet the credit policy. Status:Charged Off:A4, A5, B2, B3 and B5 are less frequent indicating 
# a lower risk with these loans. G1 and G3-G5 loans are more frequent and it could be argued that
# G class loans are higher risk of loss to the company.

# INQ LAST 6 MONTHS------------------------------------------------------------------------------
multi_chi_squared(subset_list1, "loan_status", "inq_last_6mths")
# From the chi2 test, only failed_loans 2 has a significant p value, the residuals
# will be investigated further.

chi_square_residual(failed_loans2, "loan_status", "inq_last_6mths")
# Charged Off:  From 4 to 9 inquiries the frequency of charging off is more frequent.
# Default: There are no significant residual values.
# Does not meet the credit policy. Status:Charged Off: from 0 to 2 inquiries ar highly infrequent 
# and potentially less likely. 4 to 15 and 32 inquiries is more frequent than expected and might indicate
# these loans are more likely to be charged off due to credit policy. 

# A viloin plot will reveal what the chi2 test residuals cannot fit: the desnity distribution
# of inquiries per loan status.
fig, ax = plt.subplots()
sns.violinplot(x='loan_status', y='inq_last_6mths', data=loans_analysis2, ax=ax)
ax.set_title('Violin plot of inq_last_6mths by Subset')
ax.set_xlabel('Subset')
ax.set_ylabel('inq_last_6mths')
plt.xticks(rotation=90)
plt.show()
# 10 inquiries or more are not as common but only appear for loans that do not meet the credit policy
# and are either charged off or fully paid. So many inquiries are usually red flags and pose questions
# about the customer's reliability and financial situation.

# DTI--------------------------------------------------------------------------------------------
multi_chi_squared(subset_list1, "loan_status", "dti")
# From the chi2 test, neither subset has a significant p value.
fl2.histogram("dti")
crl2 = Plotter(current_loans2)
crl2.histogram("dti")
fpl2 = Plotter(fpaid_loans2)
fpl2.histogram("dti")
# All three histograms appear to have a nromal distribution with no distinct bias in the major DTI's
# per loan status.
fig, ax = plt.subplots()
sns.violinplot(x='loan_status', y='dti', data=loans_analysis2, ax=ax)
ax.set_title('Violin plot of dti by Subset')
ax.set_xlabel('Subset')
ax.set_ylabel('dti')
plt.xticks(rotation=90)
plt.show()
# The DTI does not appear to be a strong indicator of if the loans will be a loss or not in this 
# violin plot of density.

# LOAN PURPOSE-----------------------------------------------------------------------------------
multi_chi_squared(subset_list1, "loan_status", "purpose")
# From the chi2 test, only failed_loans2 has a significant p value, the residuals
# will be investigated further.
chi_square_residual(failed_loans2, "loan_status", "purpose")
# Here we can see that for:
# Charged Off: no significant overrepresentation of any loan purpose
# Default: credit card (2.03) has a higher frequency and potentially a higher likelyhood of resulting in default
# Does not meet the credit policy. Status:Charged Off: credit card (-2.73) and debt_consolidation 
# (-3.81) are all less frequenty observed than expected and so might be less likely to result in being charged off.
# Educational (6.72), home_improvement (4.45), medical (2.03), other (3.42) and small_business (3.21) are more 
# frequent and more likely to become losses.

#INTEREST RATE-------------------------------------------------------------------
multi_chi_squared(subset_list1, "loan_status", "int_rate")
# From the chi2 test, only failed_loans2 has a significant p value, the residuals would not 
# be feasible to look over as the degrees of freedom are 772.
# Instead a violin plot will be used.
plt.figure(figsize=(10,6))
sns.violinplot(x='loan_status', y='int_rate', data=loans_analysis2)
plt.title('Violin plot of Interest Rate by Loan Status')
plt.xlabel('Loan Status')
plt.ylabel('Interest Rate')
plt.xticks(rotation = 90)
plt.show()
# Failed loans appear to have more of their density distributed towards higher interest rates. 
# But non credit policy compliant carhged off loanswhen compared to fully paid have a similar 
# density distribution to similar loans that are fully paid.

# HOME OWNERSHIP---------------------------------------------------------------------------------
multi_chi_squared(subset_list1, "loan_status", "home_ownership")
# From the chi2 test, only failed_loans2 has a significant p value, the residuals
# will be investigated further.
chi_square_residual(failed_loans2, "loan_status", "home_ownership")
# Charged Off: No significant results.
# Default: No significant results.
# Does not meet the credit policy. Status:Charged Off: None (4.19) and Other (5.32) appear to be
# more frequent than expected. Such a ownership status might indicate financial instability with
# high risk for a loss to the company.

# DELINQUENCY OVER PAST TWO YEARS----------------------------------------------------------------
multi_chi_squared(subset_list1, "loan_status", "delinq_2yrs")
# From the chi2 test, only failed_loans2 has a significant p value, the residuals
# will be investigated further.
chi_square_residual(failed_loans2, "loan_status", "delinq_2yrs")
# Charged Off: No significant results.
# Default: 3 delinquencies was more frequent than expected and 6 was even more so.
# Does not meet the credit policy. Status:Charged Off: No significant results.
# While not a very strong trend it is still worth paying attention to loans with 6+ delinquencies.

# OPEN ACCOUNTS----------------------------------------------------------------------------------
multi_chi_squared(subset_list1, "loan_status", "open_accounts")
# From the chi2 test, only failed_loans2 has a significant p value. The residuals would not 
# be feasible to look over as the degrees of freedom are 72.
# Instead a violin plot will be used.
plt.figure(figsize=(10,6))
sns.violinplot(x='loan_status', y='open_accounts', data=loans_analysis2)
plt.title('Violin plot of Open Accounts Loan Status')
plt.xlabel('Loan Status')
plt.ylabel('Open Accounts')
plt.xticks(rotation = 90)
plt.show()
# While there are not large differences in the desnity distributions there does appear to be a greater
# desnity towards 20 open accounts in defualted loans, similar to loans that are late or in the grace
# period, potentially indicating these are loans than may fail. Fully paid and current loans have a large 
# desnity around just below 10 open accounts, similar to charged off category. This may show that its'
# not a strong indicator whether a loan will charge off or not.

# TOTAL ACCOUNTS---------------------------------------------------------------------------------
multi_chi_squared(subset_list1, "loan_status", "total_accounts")
# From the chi2 test, only failed_loans2 has a significant p value. The residuals would not 
# be feasible to look over as the degrees of freedom are 132.
# Instead a violin plot will be used.
plt.figure(figsize=(10,6))
sns.violinplot(x='loan_status', y='total_accounts', data=loans_analysis2)
plt.title('Violin plot of Total Accounts Loan Status')
plt.xlabel('Loan Status')
plt.ylabel('Total Accounts')
plt.xticks(rotation = 90)
plt.show()
# As with open accounts, charged off loans share similarities with current and fully paid loans. However 
# loans that do not meet the credit policy show greater distribution of their density between 40 and
# 60 accounts. This might be something to pay attention to in late loans as grace period loans also show
# a higher distribution towards 40 accounts.
