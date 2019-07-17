import pandas as pd
import numpy as np

# Statsmodels is used for running the regression
import statsmodels.api as sm

#1 Read data
df = pd.read_csv('data.csv')
df['intercept'] = 1   # Add intercept column

#2 Linear model
linear_model = sm.OLS(df['dependent'], df[['intercept', 'independent']])
results = linear_model.fit()
results.summary()

#3 Logit model
# Make sure the dependent variable is between 0 and 1.
logit_model = sm.Logit(df['dependent'], df[['intercept', 'independent']])
results2 = logit_model.fit()
results2.summary()

# The end.
