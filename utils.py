from urllib.request import urlretrieve

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns


# preparing data
def import_data(url, file_name):
    urlretrieve(url, file_name)


def prepare_dataframe(file_name):
    df = pd.read_csv(file_name)
    print(df)
    df.info()
    medical_df_desc = df.describe()
    print(medical_df_desc)
    return df


# the following settings will improve the default style and font sizes for our charts
def set_plot_style():
    sns.set_style("darkgrid")
    matplotlib.rcParams["font.size"] = 11
    matplotlib.rcParams["figure.figsize"] = (10, 6)


# Root Mean Squared Error
def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets - predictions)))
