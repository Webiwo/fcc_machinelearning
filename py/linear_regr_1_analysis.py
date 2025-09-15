import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

import utils


# ------------------------------------------------------------------------------------------------------------
# EXPLORATORY ANALYSIS AND VISUALIZATION
# ------------------------------------------------------------------------------------------------------------
# Let's explore the data by visualizing the distribution of values in some columns of the dataset,
# and the relationships between "charges" and other columns.

#### Age
def show_age_distribution(df):
    fig = px.histogram(df,
                       x="age",
                       marginal="box",
                       nbins=47,  # 64-18
                       title="Distribution of Age")
    fig.update_layout(bargap=0.1)
    return fig


#### Body Mass Index
def show_bmi_distribution(df):
    fig = px.histogram(df,
                       x="bmi",
                       marginal="box",
                       color_discrete_sequence=["#FF0000"],
                       title='Distribution of BMI (Body Mass Index)')
    fig.update_layout(bargap=0.1)
    return fig


#### Charges
def show_charges_distribution(df, column="smoker"):
    fig = px.histogram(df,
                       x="charges",
                       marginal="box",
                       color=column,
                       color_discrete_sequence=["green", "blue", "red", "yellow"],
                       title='Annual Medical Charges')
    fig.update_layout(bargap=0.1)
    return fig


def show_charges_relationship(df, column):
    fig = px.scatter(df,
                     x=column,
                     y="charges",
                     color="smoker",
                     opacity=0.8,
                     hover_data=["age", "sex", "region", "bmi"],
                     title=f"{column.capitalize()} vs. Charges")
    fig.update_traces(marker_size=5)
    return fig


def show_children_vs_charges(df):
    fig = px.violin(df,
                    x="children",
                    y="charges",
                    hover_data=["age", "sex", "region", "bmi"],
                    title="Children vs. Charges")
    return fig


#### Smokers
def show_smoker_distribution(df):
    fig = px.histogram(df,
                       x="smoker",
                       color="sex",
                       color_discrete_sequence=["green", "blue"],
                       title="Distribution of Smoker")
    fig.update_layout(bargap=0.1)
    return fig


# ------------------------------------------------------------------------------------------------------------
# CORRELATION COEFFICIENT
# ------------------------------------------------------------------------------------------------------------
# https://www.youtube.com/watch?v=xZ_z8KWkhXE

def calculate_correlation(df):
    return df.charges.corr(df.age), df.charges.corr(df.children)


def calculate_correlation_categorical(df):
    smoker_values = {"no": 0, "yes": 1}
    smoker_num = df.smoker.map(smoker_values)
    return df.charges.corr(smoker_num)


def show_heatmap(df):
    df_encoded = pd.get_dummies(df, drop_first=True)
    corr = df_encoded.corr()
    print(corr)
    sns.heatmap(corr, cmap="Reds", annot=True)
    plt.title("Correlation Matrix")
    plt.show()


# ------------------------------------------------------------------------------------------------------------
medical_charges_url = "https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv"
file_name = "./data/charges.csv"
# utils.import_data(medical_charges_url, file_name)
medical_df = utils.prepare_dataframe(file_name)
utils.set_plot_style()
graphs = [
    show_age_distribution(medical_df),
    show_bmi_distribution(medical_df),
    show_charges_distribution(medical_df, "smoker"),
    show_charges_distribution(medical_df, "region"),
    show_charges_distribution(medical_df, "sex"),
    show_children_vs_charges(medical_df),
    show_smoker_distribution(medical_df),
    show_charges_relationship(medical_df, "age"),
    show_charges_relationship(medical_df, "bmi")
]

# graphs[5].show()
# for graph in graphs:
#    graph.show()
# for i in range(7,9):
#    graphs[i].show()

# ------------------------------------------------------------------------------------------------------------
print(calculate_correlation(medical_df))
print(calculate_correlation_categorical(medical_df))
show_heatmap(medical_df)

# -----------------------------------------------------------------------------------------------------------
# ------- Linear Regression using a Single Feature
# -----------------------------------------------------------------------------------------------------------
