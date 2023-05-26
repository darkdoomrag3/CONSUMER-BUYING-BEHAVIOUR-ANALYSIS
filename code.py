import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(
    "../input/customer-personality-analysis/marketing_data_campagin.csv", sep="\t")
df.head()
df.info()

# summary statistics of the data
df.describe()

# DATAFRAME
df.shape()

df.isna().sum()
df = df.drop(['Z_CostContact', 'Z_Revenue'], axis=1)
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
# Classification  'Married', 'Together' as "relationship" . Whereas 'Divorced', 'Widow', 'Alone', 'YOLO', 'Absurd' as "Single
df['Marital_Status'].value_counts().sort_index()
df = pd.DataFrame({'Marital_Status': ['Married', 'Single', 'Divorced']})

sns.countplot(df['Marital_Status'])
sns.set(rc={'figure.figsize': (4, 4)})
plt.show()
product_data = df[['MntWines', 'MntFruits', 'MntMeatProducts',
                   'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].values
Products_DF = pd.DataFrame(product_data, columns=[
                           'Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold'])
Products_DF.head()

# Separating Products in order to implement  Dataframe for Association Rule

product_data = df[['MntWines', 'MntFruits', 'MntMeatProducts',
                   'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].values
Products_DF = pd.DataFrame(product_data, columns=[
                           'Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold'])
Products_DF.head()
# Combining different dataframe into a single column
df['Kids'] = df['Kidhome'] + df['Teenhome']
df['Expenses'] = df[['MntWines', 'MntFruits', 'MntMeatProducts',
                     'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum(axis=1)
df['TotalAcceptedCmp'] = df[['AcceptedCmp1', 'AcceptedCmp2',
                             'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response']].sum(axis=1)
df['NumTotalPurchases'] = df[['NumWebPurchases', 'NumCatalogPurchases',
                              'NumStorePurchases', 'NumDealsPurchases']].sum(axis=1)
# reducing dimension and complexity of model
col_del = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response", "NumWebVisitsMonth", "NumWebPurchases", "NumCatalogPurchases",
           "NumStorePurchases", "NumDealsPurchases", "Kidhome", "Teenhome", "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
df = df.drop(columns=col_del, axis=1)
df.head()
# Adding column: "Age"
df['Age'] = df['Year_Birth'].apply(lambda x: 2015 - x)
df['Education'].value_counts()

# Undergrad    20
# Masters     15
# PhD         10
# High School  5
# Other       0
# Name: Education, dtype: int64

df['Education'] = df['Education'].replace({'PhD': 'PG', '2n Cycle': 'PG', 'Graduation': 'PG', 'Master': 'PG', 'Basic': 'UG'})
df['Dt_Customer'] = pd.to_datetime(df.Dt_Customer)
df['first_day'] = '01-01-2015'
df['first_day'] = pd.to_datetime(df.first_day)
df['day_engaged'] = (df['first_day'] - df['Dt_Customer']).dt.days

#Visualization
plt.rcParams.update(plt.rcParamsDefault)

fig, ax = plt.subplots(figsize=(8, 8))
sns.barplot(x=df['Marital_Status'], y=df['Expenses'], hue=df["Education"], ax=ax)
ax.set_title("Analysis of the Correlation between Marital Status and Expenses with respect to Education")
plt.show()

