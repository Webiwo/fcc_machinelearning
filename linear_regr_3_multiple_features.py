import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import utils

file_name = "charges.csv"
medical_df = utils.prepare_dataframe(file_name)


# -----------------------------------------------------------------------------------------------------------
# ------- Linear Regression using Multiple Features
# -----------------------------------------------------------------------------------------------------------

def calculate_linear_regression(df, plot=False):
    # charges = w1 × age + w2 * bmi + b
    # ------------------------------------------
    # Create inputs and targets
    X, y = df[["age", "bmi"]], df["charges"]

    # Create and train model
    model = LinearRegression()
    model.fit(X, y)

    # Generate predictions
    y_pred = model.predict(X)

    # Compute loss to evaluate the model
    loss = utils.rmse(y, y_pred)
    print(f"------ LOSS: {loss}")

    # print(f"CORR: {df.charges.corr(df.bmi)}")

    if plot:
        fig = px.scatter(df, x="bmi", y="charges", title="BMI vs. Charges")
        fig.update_traces(marker_size=5)
        fig.show()

    if plot:
        fig = px.scatter_3d(df, x="age", y="bmi", z="charges")
        fig.update_traces(marker_size=3, marker_opacity=0.5)
        fig.show()

    print(f"COEF: {model.coef_}, INTERCEPT: {model.intercept_}")

    # charges = w1 × age + w2 × bmi + w3 × children + b
    # --------------------------------------------------
    if plot:
        fig = px.strip(df, x="children", y="charges", title="Children vs. Charges")
        fig.update_traces(marker_size=5, marker_opacity=0.6)
        fig.show()

    X, y = df[["age", "bmi", "children"]], df["charges"]
    model = LinearRegression()
    model.fit(X, y)
    y_preds = model.predict(X)
    loss = utils.rmse(y, y_preds)
    print(f"------ LOSS: {loss}")
    print(f"COEF: {model.coef_}, INTERCEPT: {model.intercept_}")


# ----------------------------------------------------------------------------------------------------------
show_graph = False

non_smoker_df = medical_df[medical_df.smoker == "no"]
calculate_linear_regression(non_smoker_df, show_graph)
# ------ LOSS: 4662.3128354612945
# COEF: [266.87657817   7.07547666], INTERCEPT: -2293.6320906488727
# ------ LOSS: 4608.470405038246
# COEF: [265.2938443    5.27956313 580.65965053], INTERCEPT: -2809.297603223591

smoker_df = medical_df[medical_df.smoker == "yes"]
calculate_linear_regression(smoker_df, show_graph)
# ------ LOSS: 5722.782238884456
# COEF: [ 266.29222371 1438.09098289], INTERCEPT: -22367.449727751246
# ------ LOSS: 5718.202480524154
# COEF: [ 264.93316919 1438.72926245  198.88027911], INTERCEPT: -22556.08819649158

calculate_linear_regression(medical_df, show_graph)
# ------ LOSS: 11374.110466839007
# COEF: [241.9307779  332.96509081], INTERCEPT: -6424.804612240761
# ------ LOSS: 11355.317901125973
# COEF: [239.99447429 332.0833645  542.86465225], INTERCEPT: -6916.243347787033

if show_graph:
    fig = px.scatter(medical_df, x="age", y="charges", color="smoker")
    fig.show()


# -----------------------------------------------------------------------------------------------------------
# ------- Using Categorical Features for Machine Learning

# 1. If a categorical column has just two categories (it's called a binary category), then we can replace their values with 0 and 1.
# 2. If a categorical column has more than 2 categories, we can perform one-hot encoding.
# 3. If the categories have a natural order (e.g. cold, neutral, warm, hot), then they can be converted to numbers (e.g. 1, 2, 3, 4) preserving the order.

def calculate_linear_regression_cat(df, column_list, smoker="all"):
    '''
    :param smoker: "yes" - smokers, "no" - non-smokers, "all" - smokers and non-smokers
    '''
    if smoker == "yes" or smoker == "no":
        df = df[df["smoker"] == smoker]

    X, y = df[column_list], df["charges"]
    model = LinearRegression()
    model.fit(X, y)
    y_preds = model.predict(X)
    loss = utils.rmse(y, y_preds)
    print(f"------ LOSS CAT: {loss}")
    return model


show_graph = False

# ---------------------------------------------------------------
# charges = w1 × age + w2 × bmi + w3 × children + w4 × smoker + b
smoker_codes = {"no": 0, "yes": 1}
medical_df["smoker_code"] = medical_df["smoker"].map(smoker_codes)
if show_graph:
    sns.barplot(medical_df, x="smoker_code", y="charges")
    plt.show()

calculate_linear_regression_cat(medical_df, ["age", "bmi", "children", "smoker_code"], "all")

# ---------------------------------------------------------------
# charges = w1 × age + w2 × bmi + w3 × children + w4 × smoker + w5 × sex + b
if show_graph:
    sns.barplot(medical_df, x="sex", y="charges")
    plt.show()
sex_codes = {"female": 0, "male": 1}
medical_df["sex_code"] = medical_df["sex"].map(sex_codes)

calculate_linear_regression_cat(medical_df, ["age", "bmi", "children", "smoker_code", "sex_code"], "all")

# ONE HOT ENCODING
# ---------------------------------------------------------------
# charges = w1 × age + w2 × bmi + w3 × children + w4 × smoker + w5 × sex + w6 × region + b
if show_graph:
    sns.barplot(medical_df, x="region", y="charges", palette="Set2", hue="region")
    plt.show()

enc = preprocessing.OneHotEncoder()
enc.fit(medical_df[["region"]])
print(enc.categories_)

print(enc.transform([["northeast"],
                     ["northwest"]]).toarray())

one_hot = enc.transform(medical_df[["region"]]).toarray()
print(one_hot)
medical_df[["northeast", "northwest", "southeast", "southwest"]] = one_hot
print(medical_df)

columns = ["age", "bmi", "children", "smoker_code", "sex_code", "northeast", "northwest", "southeast", "southwest"]
calculate_linear_regression_cat(medical_df, columns, "yes")
calculate_linear_regression_cat(medical_df, columns, "no")
model = calculate_linear_regression_cat(medical_df, columns, "all")

# MODEL IMPROVEMENTS
# ---------------------------------------------------------------

# charges = w1 × age + w2 × bmi + w3 × children + w4 × smoker + w5 × sex + w6 × region + b

# values for w1-w6
w1w6 = model.coef_
print(f"W1-W6: {w1w6}")

# value of b
b = model.intercept_
print(f"B: {b}")

weights_df = pd.DataFrame({
    "feature": np.append(columns, "intercept"),
    "weight": np.append(model.coef_, model.intercept_)
})

print(weights_df)
# 0          age    256.856353
# 1          bmi    339.193454
# 2     children    475.500545

# Why BMI and the "northeast" have a higher weight than age?
# Keep in mind that the range of values for BMI is limited (15 to 40) and the "northeast" column only takes the values 0 and 1.

# Because different columns have different ranges, we run into two issues:
# 1. We can't compare the weights of different column to identify which features are important
# 2. A column with a larger range of inputs may disproportionately affect the loss and dominate the optimization process.

# Standardization - z = (x - μ) / σ
# We can apply scaling using the StandardScaler

numeric_cols = ["age", "bmi", "children"]
scaler = StandardScaler()
scaler.fit(medical_df[numeric_cols])
print(f"Scaler mean: {scaler.mean_}")
print(f"Scaler var: {scaler.var_}")

# Scale data
scaled_inputs = scaler.transform(medical_df[numeric_cols])
print(f"Scaled inputs: \n{scaled_inputs}")

cat_cols = ['smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast', 'southwest']
cat_inputs = medical_df[cat_cols]

X = np.concat((scaled_inputs, cat_inputs), axis=1)
y = medical_df["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
y_preds = model.predict(X_test)
loss = utils.rmse(y_test, y_preds)
print(f"------ LOSS SCALE: {loss}")

# We can now compare the weights in the formula:
# charges = w1 × age + w2 × bmi + w3 × children + w4 × smoker + w5 × sex + w6 × region + b

weights_df = pd.DataFrame({
    "feature": np.append(numeric_cols + cat_cols, "intercept"),
    "weight": np.append(model.coef_, model.intercept_)
})
weights_sorted_df = weights_df.sort_values("weight", ascending=False)
print(weights_sorted_df)
# 3  smoker_code  23848.534542  # 1
# 9    intercept   8466.483215
# 0          age   3607.472736  # 2
# 1          bmi   2067.691966  # 3


# ---------------------------------------------------------------
# calculate new customer charge
new_customer = [[28, 30, 2, 1, 0, 0, 1, 0, 0, 0]]
scaled_new = scaler.transform([[28, 30, 2]])
print(scaled_new)  # [[-0.79795355 -0.10882659  0.75107928]]

result = model.predict([[-0.79795355, -0.10882659, 0.75107928, 1, 0, 0, 1, 0, 0]])
print(result)  # [29875.81463371]

# https://www.kaggle.com/code/hely333/eda-regression/input
# https://www.kaggle.com/datasets/vikrishnan/boston-house-prices
# https://www.kaggle.com/datasets/budincsevity/szeged-weather
# PyTorch: https://jovian.ai/aakashns/02-linear-regression
