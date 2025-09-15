from unittest.mock import inplace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

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
    fig = px.scatter(raw_df.sample(2000), x="Temp3pm", y="Pressure3pm", color="RainTomorrow",
                     title="Temp (3 pm) vs. Pressure (3 pm)")
    fig.update_traces(marker_size=10)
    fig.show()

# ------------------------------------------------------------------------------------------------------------
# Working with a Sample
use_sample = False
sample_fraction = 0.1
if use_sample:
    raw_df = raw_df.sample(frac=sample_fraction).copy()


# ------------------------------------------------------------------------------------------------------------
# Training, Validation and Test Sets

def split_train_val_test(df):
    train_val_dfm, test_dfm = train_test_split(df, test_size=0.2, random_state=42)
    train_dfm, val_dfm = train_test_split(train_val_dfm, train_size=0.25, random_state=42)
    return train_dfm, val_dfm, test_dfm


def show_rows_per_year(df):
    plt.title("No. of rows per Year")
    sns.countplot(x=pd.to_datetime(df["Date"]).dt.year)
    plt.show()


def split_train_val_test_by_date(df, year):
    date_year = pd.to_datetime(df["Date"]).dt.year
    train_dfm = df[date_year < year]
    val_dfm = df[date_year == year]
    test_dfm = df[date_year > year]
    return train_dfm, val_dfm, test_dfm


# show_rows_per_year(raw_df)
train_df, val_df, test_df = split_train_val_test_by_date(raw_df, 2015)
print(train_df.shape, val_df.shape, test_df.shape)

# ------------------------------------------------------------------------------------------------------------
# Identifying Input and Target Columns

input_cols = list(train_df.columns)[1:-1]
target_col = "RainTomorrow"
print(f"Input Columns: {input_cols}")

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

print("======= Numeric and Categorical Columns:")
print(X_train[numeric_cols].describe())
print(X_train[categorical_cols].nunique())

# ------------------------------------------------------------------------------------------------------------
# Imputing Missing Numeric Data
# help(SimpleImputer)

print("======= Is NA sum - numeric columns:")
print(raw_df[numeric_cols].isna().sum())
imputer = SimpleImputer(strategy="mean")
imputer.fit(raw_df[numeric_cols])
# https://scikit-learn.org/stable/modules/impute.html

print("======= Imputer statistics:")
print(list(imputer.statistics_))

X_train[numeric_cols] = imputer.transform(X_train[numeric_cols])
X_val[numeric_cols] = imputer.transform(X_val[numeric_cols])
X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])
print(X_train[numeric_cols].isna().sum())

# ------------------------------------------------------------------------------------------------------------
# Scaling Numeric Features

scaler = MinMaxScaler()
scaler.fit(raw_df[numeric_cols])

print('Minimum:')
print(list(scaler.data_min_))
print('Maximum:')
print(list(scaler.data_max_))

X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
X_val[numeric_cols] = scaler.transform((X_val[numeric_cols]))
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
print("======= After MinMaxScaler")
print(X_train)
print(X_train.isna().sum())

# ------------------------------------------------------------------------------------------------------------
# Encoding Categorical Data

print('======= One Hot Encoder:')
print(X_train[categorical_cols].isna().sum())
X_train[categorical_cols] = X_train[categorical_cols].fillna("Unknown")
X_val[categorical_cols] = X_val[categorical_cols].fillna("Unknown")
X_test[categorical_cols] = X_test[categorical_cols].fillna("Unknown")
print(X_train[categorical_cols].isna().sum())

enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
train_enc = enc.fit_transform(X_train[categorical_cols])
cols_encoded = enc.get_feature_names_out(categorical_cols).tolist()
X_train_enc = pd.DataFrame(train_enc, index=X_train.index, columns=cols_encoded)

val_enc = enc.fit_transform(X_val[categorical_cols])
X_val_enc = pd.DataFrame(val_enc, index=X_val.index, columns=cols_encoded)

test_enc = enc.fit_transform(X_test[categorical_cols])
X_test_enc = pd.DataFrame(test_enc, index=X_test.index, columns=cols_encoded)

print("X_train_encoded:")
print(X_train_enc)

# ------------------------------------------------------------------------------------------------------------
# Saving Processed Data to Disk

X_train_new = pd.concat([X_train_enc, X_train[numeric_cols]], axis=1)
X_train_new.to_parquet("x_train.parquet")

X_val_new = pd.concat([X_val_enc, X_val[numeric_cols]], axis=1)
X_val_enc.to_parquet("x_val.parquet")

X_test_new = pd.concat([X_test_enc, X_test[numeric_cols]], axis=1)
X_test_enc.to_parquet("x_test.parquet")

X_train = pd.read_parquet("x_train.parquet")
X_val = pd.read_parquet("x_val.parquet")
X_test = pd.read_parquet("x_test.parquet")

print(X_train_new.compare(X_train))

# ------------------------------------------------------------------------------------------------------------
# Training a Logistic Regression Model

all_cols = numeric_cols + cols_encoded
print(all_cols)

model = LogisticRegression(solver="liblinear")
model.fit(X_train[all_cols], y_train)
print(model.coef_)

weight_df = pd.DataFrame({
    "feature": all_cols,
    "weight": model.coef_.tolist()[0]
})

# plt.figure(figsize=(10, 50))
sns.barplot(data=weight_df.sort_values("weight", ascending=False).head(20), x="weight", y="feature")
plt.show()
