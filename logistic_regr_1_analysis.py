import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

import utils

file_name = "./data/weatherAUS.csv"
raw_df = utils.prepare_dataframe(file_name)
raw_df.dropna(subset=["RainToday", "RainTomorrow"], inplace=True)
raw_df.info()

# ------------------------------------------------------------------------------------------------------------
# EXPLORATORY ANALYSIS AND VISUALIZATION
# https://www.kaggle.com/jsphyg/weather-dataset-rattle-package
# ------------------------------------------------------------------------------------------------------------
# Before training a machine learning model, it's always a good idea to explore the distributions of
# various columns and see how they are related to the target column

# raw_df["Location"].nunique()
show_graph = False
if show_graph:
    fig = px.histogram(raw_df, x="Location", title="Location vs. Rainy Days", color="RainToday")
    fig.show()

    fig = px.histogram(raw_df, x="Temp3pm", title='Temperature at 3 pm vs. Rain Tomorrow', color='RainTomorrow')
    fig.show()

    # Rain Today - imbalanced class (No=93k, Yes=17k)
    fig = px.histogram(raw_df, x='RainTomorrow', color='RainToday', title='Rain Tomorrow vs. Rain Today')
    fig.show()

    fig = px.scatter(raw_df.sample(2000), x="MinTemp", y="MaxTemp", color="RainToday", title="Min Temp. vs Max Temp.")
    fig.update_traces(marker_size=10)
    fig.show()

    fig = px.scatter(raw_df.sample(2000), x="Temp3pm", y="Humidity3pm", color="RainTomorrow",
                     title="Temp (3 pm) vs. Humidity (3 pm)")
    fig.update_traces(marker_size=10)
    fig.show()


# ------------------------------------------------------------------------------------------------------------
# Training, Validation and Test Sets

def split_train_val_test(df):
    train_val_dfm, test_dfm = train_test_split(df, test_size=0.2, random_state=42)
    train_dfm, val_dfm = train_test_split(train_val_dfm, train_size=0.25, random_state=42)
    return train_dfm, val_dfm, test_dfm


def split_train_val_test_by_date(df, year):
    date_year = pd.to_datetime(df["Date"]).dt.year
    train_dfm = df[date_year < year]
    val_dfm = df[date_year == year]
    test_dfm = df[date_year > year]
    return train_dfm, val_dfm, test_dfm


train_df, val_df, test_df = split_train_val_test_by_date(raw_df, 2015)
print(train_df.shape, val_df.shape, test_df.shape)

# ------------------------------------------------------------------------------------------------------------
# Identifying Input and Target Columns

input_cols = list(train_df)[1:-1]
target_col = "RainTomorrow"
print(input_cols)

X_train = train_df[input_cols].copy()
y_train = train_df[target_col].copy()

X_val = val_df[input_cols].copy()
y_val = val_df[target_col].copy()

X_test = test_df[input_cols].copy()
y_test = test_df[target_col].copy()

print(X_train[:10])

# Numeric and Categorical columns
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes("object").columns.tolist()

print(X_train[numeric_cols].describe())
print(X_train[categorical_cols].nunique())

# ------------------------------------------------------------------------------------------------------------
# Imputing Missing Numeric Data
help(SimpleImputer)

print(raw_df[numeric_cols].isna().sum())
imputer = SimpleImputer(strategy="mean")
imputer.fit(raw_df[numeric_cols])
print(list(imputer.statistics_))

X_train[numeric_cols] = imputer.transform(X_train[numeric_cols])
X_val[numeric_cols] = imputer.transform(X_val[numeric_cols])
X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])
print(X_train[numeric_cols].isna().sum())
