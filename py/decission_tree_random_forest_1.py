import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

import utils

file_name = "./data/weatherAUS.csv"
raw_df = utils.prepare_dataframe(file_name)
raw_df.dropna(subset=["RainToday", "RainTomorrow"], inplace=True)
raw_df.info()

# ------------------------------------------------------------------------------------------------------------
# Modeling
df = raw_df.copy()

# Numeric columns
print("**************************************************** NUMERIC COLUMNS")
numeric_cols = df.select_dtypes(include=np.number).columns.to_list()
print(f"Numeric columns: {numeric_cols}")
print(df[numeric_cols].describe())
print(df.isna().sum())

imputer = SimpleImputer(strategy="median")
imputer.fit(df[numeric_cols])

print(list(imputer.statistics_))

df[numeric_cols] = imputer.transform(df[numeric_cols])
print(df.isna().sum())

# Scaling Numeric Features
print("**************************************************** SCALING NUMERIC FEATURES")
scaler = MinMaxScaler()
scaler.fit(df[numeric_cols])
df[numeric_cols] = scaler.transform(df[numeric_cols])
print("After MinMaxScaler:")
print(df[numeric_cols][:5])

# Categorical columns
print("**************************************************** CATEGORICAL COLUMNS")
categorical_cols = df.select_dtypes("object").columns[1:-1].to_list()
print(categorical_cols)

print(df[categorical_cols].isna().sum())
df[categorical_cols] = df[categorical_cols].fillna("Unknown")
print(df[categorical_cols].isna().sum())

dummies = pd.get_dummies(df[categorical_cols], dtype=float)
print(dummies)

# Prepare final DF
all_categorical_cols = df.select_dtypes("object").columns.to_list()  # with Date
df_final = pd.concat([df["Date"], df[numeric_cols], dummies, df["RainTomorrow"]], axis=1)
print(df_final[:5])
print(df_final.isna().sum())

# Split into train, val and test
print("**************************************************** SPLIT")
date_year = pd.to_datetime(df_final["Date"]).dt.year
print(date_year)

split_year = 2015
target_column = ["RainTomorrow", "Date"]

X_train = df_final[date_year < split_year]
y_train = X_train[target_column]
X_train = X_train.drop(labels=target_column, axis=1)
print(X_train[:3])
print(X_train.shape)
print(y_train.shape)

X_val = df_final[date_year == split_year]
y_val = X_val[target_column]
X_val = X_val.drop(labels=target_column, axis=1)
print(X_val.shape)
print(y_val.shape)

X_test = df_final[date_year > split_year]
y_test = X_test[target_column]
X_test = X_test.drop(labels=target_column, axis=1)
print(X_test.shape)
print(y_test.shape)

# DecisionTreeClassifier
print("**************************************************** DecisionTreeClassifier")

model = DecisionTreeClassifier(random_state=42)

start_time = time.time()
model.fit(X_train, y_train)
print("time elapsed: {:.2f}s".format(time.time() - start_time))
