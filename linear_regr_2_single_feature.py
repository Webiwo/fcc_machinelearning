import matplotlib.pyplot as plt
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor

import utils

file_name = "charges.csv"
medical_df = utils.prepare_dataframe(file_name)

# -----------------------------------------------------------------------------------------------------------
# ------- Linear Regression using a Single Feature
# ------- y = w × x + b >>>> charges = w × age + b
# -----------------------------------------------------------------------------------------------------------

# Let's try to find a way of estimating the value of "charges" using the value of "age" for non-smokers

non_smoker_df = medical_df[medical_df.smoker == "no"]


# plt.title("Age vs. Charges")
# sns.scatterplot(data=non_smoker_df, x="age", y="charges", alpha=0.6, s=15)
# plt.show()

def estimate_charges(age, w, b):
    return w * age + b


def try_parameters(w, b):
    ages = non_smoker_df.age
    targets = non_smoker_df.charges
    predictions = estimate_charges(ages, w, b)

    plt.plot(ages, predictions, c="r", alpha=0.9)
    plt.scatter(ages, targets, s=8, alpha=0.6)
    plt.xlabel("Age")
    plt.ylabel("Charges")
    plt.show()

    loss = utils.rmse(targets, predictions)
    print(f"------ LOSS: {loss}")


# try_parameters(400, 5000)


# -----------------------------------------------------------------------------------------------------------
# ------- Linear Regression using a Scikit-learn
# -----------------------------------------------------------------------------------------------------------
# https://www.youtube.com/watch?v=szXbuO3bVRk
# https://www.youtube.com/watch?v=sDv4f4s2SB8


def age_linear_regression(model):
    # help(model.fit)

    X = non_smoker_df[["age"]]
    y = non_smoker_df.charges
    print(X.shape, y.shape)

    model.fit(X, y)
    predictions = model.predict(X)
    print(utils.rmse(y, predictions))

    # w - model.coef_
    # b - model.intercept_
    print(f"w: {model.coef_}, b: {model.intercept_}")
    try_parameters(model.coef_, model.intercept_)


model1 = LinearRegression()
age_linear_regression(model1)

model2 = SGDRegressor()
age_linear_regression(model2)

model3 = LassoLars()
age_linear_regression(model3)
