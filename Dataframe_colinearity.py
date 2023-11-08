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

with open(r"C:\Users\Dr Dankali\AICORE\EDA_Finance\exploratory-data-analysis---customer-loans-in-finance\EDA_outliers_removed.pkl", 'rb') as file:
    outlier_removed_loans = pickle.load(file)

# To test for colinearity we will first identify strongly correalted groups of variables with 
# a correlation matrix from the Plotter class

orl1 = Plotter(outlier_removed_loans)

orl1.correlation_matrix()

# From this output the following is clear:
# -Installment is strongly correlated with loan_amount, funded_amount and funded_amount_inv
# -total_rec_prncp is strongly correlated with total_payment_inv and total_payment

# These results are grounds for further investigation into colinearity using VIF scores

w = "installment"
x = "funded_amount_inv"
y = "loan_amount"
z = "funded_amount"

print(orl1.VIF(orl1, w, x, y, z))
print(orl1.VIF(orl1, x, y, z, w))
print(orl1.VIF(orl1, y, z, w, x))
print(orl1.VIF(orl1, z, w, x, y,))

# By cycling through the variables and changing whitch variable is being predicted by the other three
# we can determine which variable is best predicted by the others (highest VIF) score.
# This process was repeated with total_rec_prncp, total_payment_inv and total_payment.
# The VIF scores indicate high colinearity, which would indicate a need for dropping one of these
# variables. However, later analysis requires these variables, so it makes sense to keep them
# despite these findings. It  may  become necessary to drop them later in preparation for machine
# learning models
