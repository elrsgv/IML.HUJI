from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"
from plotly.subplots import make_subplots
import os.path

def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # Not sure in which directory the file will be and I want to support case I move the file
    # to the same directory as it is in the submmition of the tar file so I check for file location
    csv_file = f'{os.getcwd()}/{filename}'
    if os.path.isfile(f'{os.getcwd()}/{filename}'):
        csv_file = f'{os.getcwd()}/{filename}'
    elif os.path.isfile(f'{os.getcwd()}/datasets/{filename}'):
        csv_file = f'{os.getcwd()}/datasets/{filename}'
    elif os.path.isfile(f'{os.getcwd()}/../datasets/{filename}'):
        csv_file = f'{os.getcwd()}/../datasets/{filename}'
    else:
        raise FileNotFoundError()
    print(f"File of dataset to load is {csv_file}")
    df = pd.read_csv(csv_file, skiprows=0, index_col=None)

    # Now - it was best to remove lines:
    #6385     "5015000190", "20140625T000000", -690500, 5, 2, 2000, 4211, "1.5", 0, 2, 4, 7, 1280, 720, 1908, 0, "98112", 47.6283, -122.301, 1680, 4000
    #10818    "1223039235",, 605000, 5, 2.75, 2910, 13332, "2", 0, 0, 4, 8, 2910, 0, 1940, 1991, "98146", 47.4977, -122.359, 1760, 8900
    #14902    "nan", "20141203T000000", 430000, 3, 2.25, 1830, 19965, "1", 0, 0, 3, 8, 1400, 430, 1976, 0, "98072", 47.7412, -122.088, 1830, 17250
    #16656    , "20140605T000000", 355500, 3, 2.5, 2600, 5540, "2", 0, 0, 3, 8, 2600, 0, 2004, 0, "98038", 47.3446, -122.041, 2600, 5540
    #20385    "1222029064", "20140626T000000", nan, 3, 1.75, 1444, 249126, "1.5", 0, 0, 3, 7, 1444, 0, 2008, 0, "98070", 47.4104, -122.486, 1760, 224770
    #20673    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    #20674    0, 0, 0, 0, 0, 0, 0, 0, 0,
    # But instead we'll filter the zero of negative lines and leave the rest.
    # In the future I'll try do filter it by np.isna to to filter nan

    # This time split and analysis was done only when I worked on hacked th
    # df["time"] = df["date"].str.split("T").str.get(1) - Not needed. no hour marked in DataSet.
    # df["date"] = df["date"].str.split("T").str.get(0).astype(int) # without the hourly time
    ## make date approximation of day since 0 B.C.
    #df["date"] = 1 * (df["date"] % 100) + 31 * ((df["date"] // 100) % 100) + 366 * ((df["date"] // 1e4))

    return df

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    def pearson_correlation(a: pd.Series,b: pd.Series) -> float:
        a, b = a.astype(float), b.astype(float)
        ab_cov_vec = (a-a.mean()) * (b-b.mean())
        ab_cov = ab_cov_vec.mean()
        ab_stds = a.std()*b.std()
        if ab_stds==0: return None
        return round(ab_cov/ab_stds,3)

    features_count = len(X.columns)
    if features_count > 4 : # print one figure for each of the properties versus price, by many subplots
        sub_rows = int(np.floor(np.sqrt(len(X.columns))/1.1).item())
        sub_cols = int(np.ceil(len(X.columns)/sub_rows).item())
        fig = make_subplots(sub_rows, sub_cols, subplot_titles=list(X.columns))
        for i, key in enumerate(X.columns):
            r = i // sub_cols + 1,
            c = i % sub_cols + 1
            pearson = pearson_correlation(y, X[key])
            print(" Pearson correlation of {} to the price is {}".format(key, pearson))
            fig.add_traces([go.Scatter(x=y, y=X[key], mode="markers", name=key,
                                   marker=dict(color="black", opacity=.7), showlegend=False)], rows=r, cols=c)
            fig["layout"]["annotations"][i]["text"] = r"$\text{{{}}}, \rho(Pearson)={}$".format(key, pearson)
            fig["layout"]["annotations"][i]["font"] = dict(size=18)
            #fig["layout"]["annotations"][i]["ax"] = r"$\text{price}$"
            #fig["layout"]["annotations"][i]["ay"] = r"$\text{{{}}}$".format(key)

        fig.update_layout(title=r"$\text{(Ex2 Practical Q1) X-axes is all graphs is the price }$", margin=dict(l=120) )
        fig.write_image(f"{output_path}/P1_feature_evaluation_{features_count}.png", scale=1, width=2000, height=1024)
        print(f"Finished writing graphs:  {output_path}/P1_feature_evaluation_{features_count}.png")

    else: # print 1 by 1 each graph for each feature.
        for i, key in enumerate(X.columns):
            pearson = pearson_correlation(y,X[key])
            print(" Pearson correlation of {} to the price is {}".format(key, pearson))
            fig = go.Figure([go.Scatter(x=y, y=X[key], name=key, showlegend=False, mode="markers",
                                        marker=dict(color="black", opacity=.7))],
                    layout=go.Layout(title=r"$\text{{{}}}, \rho(Pearson)={}$".format(key, pearson),
                                     xaxis={"title": "x:  price"},
                                     yaxis={"title": "y:  {}".format(key)})
            )
            fig.write_image(f"{output_path}/P1_feature_evaluation_{key}.png", scale=1, width=800, height=600)
            print(f"Finished writing graphs:  {output_path}/P1_feature_evaluation_{key}.png")



if __name__ == '__main__':
    np.random.seed(0)

    print(" # Question 1 - Load and preprocessing of housing prices dataset")
    df = load_data("house_prices.csv")
    df = df.drop(df[df["price"] <= 0].index, axis=0)
    #df = df[:500] # temporary
    price = df["price"]
    df["yr_new_renew"] = df[["yr_built", "yr_renovated"]].max(axis=1) # new built-up is good as renovated.
    #Making polynomial long and lat to make center of location finding.
    df["long2"] = df["long"]**2
    df["lat2"] =  df["lat"]**2
    df["long_lat"] = df["long"] * df["lat"]
    df["long2_lat"] = df["long"]**2 * df["lat"]
    df["long_lat2"] = df["long"] * df["lat"]**2
    df["long2_lat2"] = df["long"]**2 * df["lat"]**2
    print("Total records found {0}".format(len(df)))
    print("Different zipcode values are {0}".format(df["zipcode"].nunique()))
    print("Different id values are {0}".format(df["id"].nunique()))
    print()

    print(" # Question 2 - Feature evaluation with respect to response (price) w/ Pearson correlation")
    del df["price"] # exclude so it will be compared to it
    df = df.drop(["date", "id"], axis=1)
    feature_evaluation(df, price)
    feature_evaluation(df[["grade", "long"]], price)
    print()

    print(" # Q1 cont. - Make zip code a one-hot parameters set")
    zipcodes_1hot = pd.get_dummies(df["zipcode"])
    df = df.join(zipcodes_1hot)

    print(" # Q1 cont. - Summary of features statistics and other data")
    df_sum = pd.DataFrame({'sum': df.sum(), 'mean': df.mean(), 'stddiv': df.std(), \
                           'n-uniq': df.nunique(), 'min': df.min(), 'max': df.max(), 'dtype': df.dtypes})
    print(df_sum.to_string())
    df = df.drop(["zipcode", "yr_renovated"], axis=1) # all that are not from fit. doubt on "lat", "long",

    print(" # Question 3 - Split samples into training- and testing sets.")
    train_frac = 0.75
    train_X, train_y, test_X, test_y = split_train_test(df, price, train_frac)
    print(" Train on {}% i.e. train @ {}, test @ {} samples".format(100*train_frac,len(train_y),len(test_y)))
    model = LinearRegression(include_intercept=True)
    model.fit(train_X, train_y)
    # Bonus check of loss for trial to improve
    train_loss, test_loss  =  model.loss(train_X,train_y), model.loss(test_X,test_y)
    print(" Bonus check of MSE loss - train: {}k, test {}k".format(round(train_loss*1e-3,1),round(test_loss*1e-3,1)))

    # we split to 75% and 25% first (in Q3). then we take 10% of the 75% as training data and test
    # against the 25% test we split earlier, continuing with 11% of the 75%, etc
    print(" # Question 4 - Fit model over increasing percentages of the overall training data")
    losses_means, losses_stds = np.empty(0), np.empty(0) # arrays for data collection
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    pcts = list(range(10, 101))
    for train_pct in pcts:
        if (train_pct % 5) == 0:  print("...{}".format(train_pct), end="") # Track run-time progress
        losses = np.empty(0) # array for specific precentage point data collection.
        for _ in range(10):
            # 1) Sample p% of the overall training data
            train_X_parly, train_y_partly, _, _ = split_train_test(train_X, train_y, train_pct / 100)
            # 2) Fit linear model (including intercept) over sampled set
            model = LinearRegression(include_intercept=True)
            model.fit(train_X_parly, train_y_partly)
            # 3) Test fitted model over test set
            predict_y = model.predict(test_X)
            # 4) Store average and variance of loss over test set
            losses = np.append(losses, model.loss(test_X,test_y))
            # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
        losses_means = np.append(losses_means, np.mean(losses))
        losses_stds = np.append(losses_stds, np.std(losses))
    fig = go.Figure([go.Scatter(x=pcts, y=losses_means, mode="markers+lines", name="Losses Mean", line=dict(dash="dash"),
                                marker=dict(color="green", opacity=.7)),
                     go.Scatter(x=pcts, y=losses_means-2*losses_stds, fill=None, mode="lines",
                                line=dict(color="lightgrey"), showlegend=False),
                     go.Scatter(x=pcts, y=losses_means+2*losses_stds, fill='tonexty', mode="lines",
                                line=dict(color="lightgrey"), showlegend=False)],
                     layout=go.Layout(title="MSE loss with mean and std-div with respect to increasing training set size",
                                     xaxis = {"title": "x: size of training data set as % of the 75% dedicated for train"},
                                     yaxis = {"title": "y: Calculate MSE loss on 25% dedicated for test"})
    )
    fig.write_image("P1Q4_loss_versus_training_set_size.png", scale=1, width=800, height=600)
    print("Finished writing graphs:  P1Q4_loss_versus_training_set_size.png")
    print()

    print("Quiz add-ons:")
    from IMLearn import metrics
    y_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
    y_pred = np.array(
        [199000.37562541, 452589.25533196, 345267.48129011, 345856.57131275, 563867.1347574, 395102.94362135])
    print("Q2 in quiz: {}".format(round(metrics.mean_square_error(y_true, y_pred),3)))

    print(" End of house_price_prediction.py ")
