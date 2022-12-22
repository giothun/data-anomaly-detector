import numpy as np
import pandas as pd
from prophet import Prophet

class Anomaly:
    def __init__(self, type_of, where, probability, anomaly):
        self.type_of = type_of
        self.where = where
        self.probability = probability
        self.anomaly = anomaly

    def __str__(self):
        print(f'Anomaly: {self.type_of} detected in {self.where} probability class {self.probability}\n{self.anomaly}')


def read_data(path, type_of_data):
    if type_of_data == "csv":
        return pd.read_csv(path)


def format_time(df, time_columns):
    for col in time_columns:
        df[col] = pd.to_datetime(df[col])


def find_duplicated(df):
    if df.duplicated().sum() > 0:
        return [Anomaly("Duplicate", "data", "A class", df[df.duplicated()])]
    else:
        return "No duplicates"


def find_outliers_in_num_data(df, num_columns):
    anomaly_arr = []
    for col in num_columns:
        cur_avg = df[col].mean()
        cur_std = df[col].std()
        a = Anomaly("stat_outlier", col, "A class",
                    df[(df[col] < cur_avg - 3 * cur_std) | (df[col] > cur_avg + 3 * cur_std)])
        anomaly_arr.append(a)
    return anomaly_arr


def find_outliers_in_num_data_prophet(df, date_col):
    for time,feature in date_col:
        cur_df = df[[time,feature]]
        cur_df.rename({time: "ds", feature: "y"}, axis=1, inplace=True)
        m = Prophet()
        m.fit(cur_df)
        forecast = m.predict(cur_df)
        return forecast


def start(df,date_col,num_col):
    format_time(df,date_col)
    find_duplicated(df)
    find_outliers_in_num_data(df,num_col)
    #find_outliers_in_num_data_prophet()


data = read_data("nyc_taxi_trip_duration.csv", "csv")