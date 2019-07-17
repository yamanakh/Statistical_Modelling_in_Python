# This code sample highlights the approach to checking a regression model for multicollinearity
# For this we will calculate Variance Inflation Factors (VIF). See also https://etav.github.io/python/vif_factor_python.html

#1 Package Import

# Standard packages
import pandas as pd
import numpy as np

# Seanborn's Pairplot in seaborn can show us pairwise relationships for all of the quantitative, explanatory variables
import seaborn as sns 

# Dmatrices is needed to calculate the VIFs
from patsy import dmatrices

# Statsmodels for the regression modelling
import statsmodels.api as sm;
from statsmodels.stats.outliers_influence import variance_inflation_factor

# For visualizations inside a Jupyter Notebook
%matplotlib inline


#2 Read data
df = pd.read_csv('data.csv')

#3 Utilize the pairplot to visualize any potential issue with multicollinearity
# ("Are several variables strongly correlated with each other?")

#4 Run a regression model
df['intercept'] = 1   # Setting up the intercept (not automatically done in statsmodels)

linear_model = sm.OLS(df['dependent'], df[['colnames', 'of', 'independents']])
results = linear_model.fit()
results.summary()   # Show the results summary

#5 Calculate the VIFs
y, X = dmatrices('dependent ~ colnames + of + independents', df, return_type='dataframe')

vif = pd.DataFrame()
vif['vif_factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['features'] = X.columns

# Rule of thumb with VIFs: Remove any one variable from a pair if the VIF values are above 10.
