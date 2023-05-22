import numpy as np                               
import pandas as pd                               
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("../input/customer-personality-analysis/marketing_campaign.csv", sep="\t")
df.head()
df.info()

#summary statistics of the data
df.describe()

#DATAFRAME
df.shape()

df.isna().sum()
df = df.drop(['Z_CostContact', 'Z_Revenue'],axis=1)
df.head()
plt.rcParams.update(plt.rcParamsDefault)

# Create a figure and set the size
fig = plt.figure(figsize=(18, 12))

# Create a heatmap of the correlation matrix
sns.heatmap(df.corr(), annot=True)

# Show the plot
plt.show()

# Create a figure and set the size
fig, ax = plt.subplots(figsize=(18, 12))

# Create a heatmap of the correlation matrix
sns.heatmap(df.corr(), annot=True, ax=ax)

# Show the plot
plt.show()

def fill_na_with_mean(df, column):
  """Fills missing values in a column with the mean of that column.

  Args:
    df: A pandas DataFrame.
    column: The name of the column to fill.

  Returns:
    A pandas DataFrame with the missing values filled.
  """
  df[column] = df[column].fillna(df[column].mean())
  return df


df = fill_na_with_mean(df, 'Income')

# Check if there are any remaining missing values.
if df.isna().any():
  print('There are still missing values in the DataFrame.')
else:
  print('All missing values have been filled.')

  df.head()
def get_marital_status_counts(df):
  """Gets the counts of each marital status in a DataFrame.

  Args:
    df: A pandas DataFrame.

  Returns:
    A pandas Series with the counts of each marital status.
  """
  return df['Marital_Status'].value_counts()


marital_status_counts = get_marital_status_counts(df)

print(marital_status_counts)

def plot_marital_status_counts(df):
  """Plots the counts of each marital status in a DataFrame.

  Args:
    df: A pandas DataFrame.

  Returns:
    A matplotlib figure.
  """
  fig, ax = plt.subplots(figsize=(4, 4))
  sns.countplot(df['Marital_Status'], ax=ax)
  return fig


fig = plot_marital_status_counts(df)

plt.show()