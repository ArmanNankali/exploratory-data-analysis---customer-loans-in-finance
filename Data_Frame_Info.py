import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pickle
import pandas as pd
import csv as csv
import numpy as np
from scipy import stats

# Create a new dataframe from the .pkl file with transformed columns
with open(r"C:\Users\Dr Dankali\AICORE\EDA_Finance\exploratory-data-analysis---customer-loans-in-finance\EDA_dtype_transformed.pkl", 'rb') as file:
    transformed_loans = pickle.load(file)

# This class allows us to obtain useful information from the dataframe
class DataFrameInfo():
    def __init__(self, df):
        self.df = df
    
    def data_type(self):
        return self.df.dtypes

    def describe(self):
        print(self.df.describe())

    def unique_categories(self):
        for col in self.df:
            unique_values = {}
            if self.df[col].dtypes.name == "category":  
                unique_values[col] = list(self.df[col].unique())
                print(f"{col} column has the following unique values {unique_values}")
            
    
    def data_shape(self):
        print(self.df.shape)
    
    def null_percent(self):
        for col in self.df:
            null_percents = {}
            try:
                null_percentage = ((self.df[col].isnull().sum()/len(self.df[col]))*100).round(2)
                null_percents[col] = null_percentage
                if null_percentage > 0:
                    print(f"{col} contains {null_percentage}% null values")
    
            except ZeroDivisionError:
                pass

# We instantiate the class and create an object
tl1 = DataFrameInfo(transformed_loans)

tl1.data_type()
tl1.describe()
tl1.unique_categories()
tl1.data_shape()
tl1.null_percent()

# We have obtained the following variables with missing (null) values
#funded_amount contains 5.54% null values
#term contains 8.8% null values
#int_rate contains 9.53% null values
#employment_length contains 3.91% null values
#mths_since_last_delinq contains 57.17% null values
#mths_since_last_record contains 88.6% null values
#last_payment_date contains 0.13% null values
#next_payment_date contains 60.13% null values
#last_credit_pull_date contains 0.01% null values
#collections_12_mths_ex_med contains 0.09% null values
#mths_since_last_major_derog contains 86.17% null values



# Here we introduce the Plotter class which allows us to easily create various plots for our data



class Plotter():
    def __init__(self, df):
        self.df = df
            
    def qq_plot(self, col):
        self.col = col
        self.qq_plot = qqplot(self.df[self.col] , scale=1 ,line='q')
        pyplot.show()
    
    def histogram(self, col):
        self.col = col
        plt.hist(self.df[self.col].dropna(), bins=30, edgecolor='black')
        plt.xlabel(self.col)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {self.col}")
        plt.show()
    
    def boxplot(self, col):
        self.col = col
        sns.boxplot(self.df[self.col].dropna())
        plt.xlabel(self.col)
        plt.title(f"Box Plot of {self.col}")
        plt.show()
    
    def correlation_matrix(self):
        numerical_df = self.df.select_dtypes(include=['int64', 'float64'])
        corr = numerical_df.corr()
        mask = np.zeros_like(corr, dtype=np.bool_)
        mask[np.triu_indices_from(mask)] = True
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, 
            square=True, linewidths=1, annot=False, cmap=cmap)
        plt.yticks(rotation=0)
        plt.title('Correlation Matrix of all Numerical Variables')
        plt.show()
    
    def correlation_matrix_no_nulls(self):
    # Create a temporary copy of the DataFrame
        temp_df = self.df.copy()

    # Select only numerical columns
        numerical_df = temp_df.select_dtypes(include=['int64', 'float64'])

    # Drop rows with null values
        numerical_df = numerical_df.dropna()

    # Calculate correlation matrix
        corr = numerical_df.corr()

    # Create a mask for the upper triangle of the correlation matrix
        mask = np.zeros_like(corr, dtype=np.bool_)
        mask[np.triu_indices_from(mask)] = True

    # Create a diverging color palette for the heatmap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Create the heatmap
        sns.heatmap(corr, mask=mask, square=True, linewidths=1, annot=False, cmap=cmap)

        plt.yticks(rotation=0)
        plt.title('Correlation Matrix of all Numerical Variables')
        plt.show()

    def kde(self, col):
        self.col = col
        sns.kdeplot(self.df[self.col].dropna(), shade=True)
        plt.xlabel(self.col)
        plt.ylabel('Density')
        plt.title(f'KDE Plot of {self.col}')
        plt.show()
    
    def dak2(self, col):
        self.col = col
        stat, p = stats.normaltest(self.df[col], nan_policy='omit')
        print(f"{self.col}")
        print("Statistics=%.3f, p=%.3f" % (stat, p))

    def linear_regression(self, predicted, *predictors):
        self.predicted = predicted
        self.predictors = predictors
        predictors_str = " + ".join(predictors)
        formula = f"{predicted} ~ {predictors_str}"
        # Drop any rows with NaN values in the specified columns
        self.df = self.df.dropna(subset=[predicted, *predictors])
        self.model = smf.ols(formula, self.df).fit()
        print(self.model.summary())
        return self.model

    def skew_plot(self, col):
        self.col = col
        self.df[self.col].hist(bins=50)
        print(f"Skew of {self.col} column is {(self.df[self.col].skew()).round(2)}")
    
    def z_score(self, col):
        self.col = col
        mean_col = np.mean(self.df[col])
        std_col = np.std(self.df[col])
        zscores = (self.df[col] - mean_col) / std_col
        outliers = (zscores > 2.5) | (zscores < -2.5)
        print(self.df[self.col][outliers])
        print(f"{self.col} number of outliers: {self.df[self.col][outliers].count()}")
        print(f"{self.col} % outliers: {(self.df[self.col][outliers].count()/len(self.df))*100}")
        return outliers
    
    def VIF(self, df_object, predicted, *predictors):
        self.predicted = predicted
        self.predictors = predictors
        self.df_object = df_object
        self.regression = self.df_object.linear_regression(self.predicted, *self.predictors)
        r2 = self.regression.rsquared
        VIF_value = 1/(1-r2)
        return f"{self.predicted} VIF value: {VIF_value}"


    
class All_Plots(Plotter):
    def all_plots(self, col_name):
        super().qq_plot(col_name)
        super().histogram(col_name)
        super().boxplot(col_name)
        super().kde(col_name)
        super().dak2(col_name)


# DROP VARIABLE: these variables have too many missing values for imputation or to drop the rows
# This is done with the knowledge that columns pertaining to missed payments likely contain so many nulls because most customers generally don't miss their payments.
# Such information could be useful but it is not poissible to now for sure what explanation there is for the nulls.
# mths_since_last_major_derog contains 86.17% null values DROP var
# mths_since_last_record contains 88.6% null values DROP var
# next_payment_date contains 60.13% null values DROP var
# mths_since_last_delinq contains 57.17% null values DROP var
transformed_loans.drop(columns=["mths_since_last_major_derog"], inplace=True)
transformed_loans.drop(columns=["mths_since_last_record"], inplace=True)
transformed_loans.drop(columns=["next_payment_date"], inplace=True)
transformed_loans.drop(columns=["mths_since_last_delinq"], inplace=True)


# DROP ROWS: The Nulls from the following variables comprise approximately 0.24% of the total data.
# Hence, the rows can be dropped without affecting the total dataset much.
# last_credit_pull_date contains 0.01% null values
# collections_12_mths_ex_med contains 0.09% null values
# last_payment_date contains 0.13% null values

transformed_loans.dropna(subset=["last_credit_pull_date"], inplace=True)
transformed_loans.dropna(subset=["collections_12_mths_ex_med"], inplace=True)
transformed_loans.dropna(subset=["last_payment_date"], inplace=True)


# With the following variables we will need to impute the null data.
#funded_amount contains 5.54% null values
#term contains 8.8% null values
#int_rate contains 9.53% null values
#employment_length contains 3.91% null values

# Funded_amount and funded_amount_inv are likely to have some correlation, so we can check this.
tl2 = Plotter(transformed_loans)
tl2.linear_regression("funded_amount", "funded_amount_inv")
# The R^2 value is 0.965 meaning they are very strongly correlated.
# From the output we can also see that the constant coefficient/intercept is 619.8326 and the coefficient for funded_amount_inv is 0.9734
# This means that:
# -when funded_amount_inv is 0, funded_amount would be 619.8326
# -for each unit increase in funded_amount_inv, funded_amount increases by 0.9734 units
# using this informationb we impute the null values in funded_amount

const_coef = 619.8326
installment_coef = 0.9734
# Identify rows with null 'funded_amount'
null_rows = transformed_loans[transformed_loans["funded_amount"].isnull()]
# Calculate predicted 'funded_amount' using regression coefficients
predicted_funded_amount = const_coef + installment_coef * null_rows["installment"]
# Impute null values with predicted 'funded_amount'
transformed_loans.loc[null_rows.index, "funded_amount"] = predicted_funded_amount


# As for term, when we visualise the data we see that all of the data pints are either 36 months or 60 months
# We will impute the null values with the mode "36"
#tl2.histogram("term")

null_rows = transformed_loans[transformed_loans["term"].isnull()]
transformed_loans.loc[null_rows.index, "term"] = "36 months"

# With the int_rate variable it was not possible to find another variable which correlated strongly with it
# Here we will impute the null data with the mean: 13.507936
transformed_loans["int_rate"].describe()

null_rows = transformed_loans[transformed_loans['int_rate'].isnull()]
transformed_loans.loc[null_rows.index, "int_rate"] = 13.507936

# As for employment_length, we can see with another histogram that the modal value is 10+ years
# We will use this value to impute the null data points
tl2.histogram("employment_length")
transformed_loans["employment_length"].describe()
null_rows = transformed_loans[transformed_loans['employment_length'].isnull()]
transformed_loans.loc[null_rows.index, "employment_length"] = "10+ years"

# Now we can see there are no variables containing null values left!
tl1.null_percent()

# For the sake of readability and memory management we will save the current state of the dataframe 
# to another .pkl file to be opened in the Dataframe_skew_and_outliers.py file
with open(r"C:\Users\Dr Dankali\AICORE\EDA_Finance\exploratory-data-analysis---customer-loans-in-finance\EDA_nulls_removed.pkl", 'wb') as file:
    pickle.dump(transformed_loans, file)




