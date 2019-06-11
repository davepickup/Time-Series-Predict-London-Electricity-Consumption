import pandas as pd
from fbprophet import Prophet


class EnergyModel:

    def __init__(self):

        self.model = None

    def preprocess_training_data(self, df):

        df['day'] = pd.to_datetime(df['day'])
        df['day'] = pd.DatetimeIndex(df['day'])
        df_prepared = df.rename(columns={'day': 'ds', 'consumption': 'y'})
        return df_prepared, None

    def fit(self, X, y):

        forecast_model = Prophet()

        forecast_model.fit(X)
        self.model = forecast_model

    def preprocess_unseen_data(self, df):

        df['day'] = pd.to_datetime(df['day'])
        df['day'] = pd.DatetimeIndex(df['day'])
        df_prep = df.rename(columns={'day': 'ds'})
        return df_prep

    def predict(self, X):

        preds = self.model.predict(X)
        return preds['yhat']
