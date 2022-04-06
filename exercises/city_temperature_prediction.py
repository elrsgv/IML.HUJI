import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

###from sklearn.pipeline import Pipeline
###from sklearn.linear_model import LinearRegression
###from sklearn.preprocessing import PolynomialFeatures
###from sklearn.model_selection import train_test_split
###split_train_test = train_test_split # alias of function when checking with sklearn

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
import os.path
import datetime
import plotly.graph_objects as go
from utils import *


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    ##filename = "City_Temperature.csv"
    if os.path.isfile(f'{os.getcwd()}/{filename}'):
        csv_file = f'{os.getcwd()}/{filename}'
    elif os.path.isfile(f'{os.getcwd()}/datasets/{filename}'):
        csv_file = f'{os.getcwd()}/datasets/{filename}'
    elif os.path.isfile(f'{os.getcwd()}/../datasets/{filename}'):
        csv_file = f'{os.getcwd()}/../datasets/{filename}'
    else:
        raise FileNotFoundError()
    print(f"File of dataset to load is {csv_file}")
    df = pd.read_csv(csv_file, skiprows=0, index_col=None, nrows=None, parse_dates=["Date"], verbose=True)
    # Add Day-Of-Year as requested in Q2:
    df["DayOfYear"] = df["Date"].map(lambda t: t.timetuple().tm_yday)
    # All Temp seems on Celsius. Filtering out many bad samples that are on negative -72 deg
    bad_min_temp = df["Temp"].min()
    print("Found bad low tempertures of {} at {} out of {} samples".format(
            bad_min_temp, len(df[df["Temp"]==bad_min_temp]), len(df)))
    df = df.drop(df[df["Temp"] == bad_min_temp].index, axis=0)
    print(df.dtypes)

    print(" Some statistics on the numerical fields data")
    dfnum = df[["Year", "Month", "Day", "Temp", "DayOfYear"]]
    df_sum = pd.DataFrame({'sum': dfnum.sum(), 'mean': dfnum.mean(), 'stddiv': dfnum.std(), \
                           'n-uniq': dfnum.nunique(), 'min': dfnum.min(), 'max': dfnum.max(), 'dtype': dfnum.dtypes})
    print(df_sum.to_string())

    #df = df[:500] # temporary
    return df


if __name__ == '__main__':
    np.random.seed(0)

    print(" # Question 1 - Load and preprocessing of city temperature dataset")
    df = load_data("City_Temperature.csv")
    print(" Example of 3 entries (transposed) of the data-frame")
    print(df.sample(3).T)
    print()


    print(" # Question 2 - Exploring data for specific country")
    print(" Adding DayOfYear was done in th load_data and processing stage.")
    dfi = df[df["Country"]=="Israel"] # DataFrame Israel
    fig = go.Figure(
        [go.Scatter(x=dfi["DayOfYear"], y=dfi["Temp"], mode="markers", name="Temperature",
                    marker=dict(color=dfi["Year"], opacity=.7), showlegend=True)],
        layout=go.Layout(title="Temperatures in Israel over the years' days over the years 1995-2020",
                         xaxis={"title": "x: Day-of-Year"},
                         yaxis={"title": "y: Temperature"})
        )
    fig.write_image("P2Q2a_Israel_temperatures_along_year.png", scale=1, width=800, height=600)
    print("Finished writing graphs:  P2Q2a_Israel_temperatures_along_year.png")
    print()
    print(" # Question 2 b - The per ‘Month’ standard-deviation of temperatures")
    dfims = dfi.groupby("Month").agg("std")  # DataFrame Israel - Monthly std-dev
    fig = go.Figure(
        go.Bar(x=dfims.index, y=dfims["Temp"], name="Temp standard-deviation", showlegend=False),
        layout=go.Layout(title="Per month standard-deviation of temperatures in Israel (over the years 1995-2020)",
                         xaxis={"title": "x: Month"},
                         yaxis={"title": "y: Temperature standard-deviation"})
        )
    fig.write_image("P2Q2b_Israel_Temperatures_STD_DEV.png", scale=1, width=800, height=600)
    print("Finished writing graphs:  P2Q2b_Israel_Temperatures_STD_DEV.png")
    print()

    print(" # Question 3 - Exploring differences between countries")
    # making the aggregated data-set with mean and std-dev for the Temperature
    df3 = pd.DataFrame.from_records(df, columns=["Country", "Month", "Temp"]).groupby(["Country", "Month"]).agg(
        ("mean", "std")).reset_index()
    df3 = df.groupby(["Country", "Month"]).agg(("mean", "std")).reset_index()
    df3 = pd.DataFrame.from_records(df,
            columns=["Country", "Month", "Temp"]).groupby(["Country", "Month"]).agg(("mean", "std")).reset_index()
    df3["Temp_mean"] = df3["Temp"]["mean"]
    df3["Temp_std"] = df3["Temp"]["std"]

    # plotting
    fig = px.line(df3, x="Month", y="Temp_mean", color="Country", error_y="Temp_std",
                  title="Temperatures at average and std-dev per month per country")
    fig.write_image("P2Q3_Temperatures_in_4_countries.png", scale=1, width=800, height=600)
    print("Finished writing graphs:  P2Q3_Temperatures_in_4_countries.png")
    print()

    print(" # Question 4 - Fitting model for different values of `k`")
    X = np.array(dfi["DayOfYear"]) # sampling on days of year
    y = np.array(dfi["Temp"]) # result of temperature per sample
    train_X, train_y, test_X, test_y = split_train_test(X, y) # 75% train
    losses = []
    for k in range(1,11):
        model = PolynomialFitting(k=k)
        model.fit(train_X, train_y)
        #print(" train average MSE loss is {}".format(round(model.loss(train_X, train_y), 3)), end="")
        print("   test average MSE loss on month {} is {}".format(k,round(model.loss(test_X, test_y), 3)))
        #losses.append(format(round(model.loss(test_X, test_y), 2)))
        losses.append(model.loss(test_X, test_y))
        if 0: # Bonus check I make for sanity check.
            predict = model.predict(dfi["DayOfYear"])
            fig = go.Figure(
                [go.Scatter(x=dfi["DayOfYear"], y=dfi["Temp"], mode="markers", name="Temperature",
                            marker=dict(color=dfi["Year"], opacity=.7), showlegend=True),
                 go.Scatter(x=dfi["DayOfYear"], y=predict, mode="markers", name="Temperature",
                            marker=dict(color="black"), showlegend=True)
                ],
            )
            fig.write_image("P2Q4_Test_model_predict_at_k_{}.png".format(k), scale=1, width=800, height=600)
            print("Finished writing graphs:  P2Q4_Test_model_predict_at_k_{}.png".format(k))
    # Figure of loss at every polynomial level
    fig = go.Figure(
        go.Bar(x=list(range(1, 11)), y=losses, name="Polynomial level fit", showlegend=False),
        layout=go.Layout(title="MSE of fitting Temp over day-of-year to polynomial - Per polynomial order 'k'",
                         xaxis={"title": "x: k - polynomial order"},
                         yaxis={"title": "y: MSE loss"})
    )
    fig.write_image("P2Q4_Israel_Polynomial_level_fit.png", scale=1, width=800, height=600)
    print("Finished writing graphs:  P2Q4_Israel_Polynomial_level_fit.png")
    print()


    print(" # Question 5 - Evaluating fitted model on different countries")
    df_losses = pd.DataFrame(columns=["Country", "loss"])
    #X = np.array(dfi["DayOfYear"]) # sampling on days of year    at Israel
    #y = np.array(dfi["Temp"]) # result of temperature per sample at Israel
    model = PolynomialFitting(k=3)
    model.fit(dfi["DayOfYear"], dfi["Temp"])  # = (X, y)
    for country in ["Israel", "Jordan", "The Netherlands", "South Africa"]:
        dfc = df[df["Country"]==country] # data-frame of country
        loss = model.loss(dfc["DayOfYear"], dfc["Temp"])
        df_losses = pd.concat([ df_losses, pd.DataFrame({ "Country": [country], "loss": [loss] }) ])
    df_losses.reset_index(inplace=True,drop=True)
    print(df_losses)
    fig = px.bar(df_losses, x="Country", y="loss", title="Losses per country when fitted model on Israel")
    fig.write_image("P2Q5_loss_at_country_w_fitted_Israel.png", scale=1, width=800, height=600)
    print("Finished writing graphs:  P2Q5_loss_at_country_w_fitted_Israel.png")
    print()
    print(" End of city_temperature_prediction.py ")
