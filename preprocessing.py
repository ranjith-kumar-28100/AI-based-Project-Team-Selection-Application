import pandas as pd
import statistics
from sklearn.preprocessing import MinMaxScaler
import logging
from datetime import datetime

logging.basicConfig(filename=f"preprocessor_{datetime.today().strftime('%Y%m%d')}.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s", filemode='a')


class DataSet:
    def __init__(self, data):
        self.data = data.copy()
        self.columns = data.columns
        self.size = data.shape[0]
        logging.info("Successfully Created Instance for the class")

    def handle_missing(self):
        for col in self.columns:
            if self.data[col].dtype == object and self.data[col].isnull().sum() > 0:
                self.data[col].fillna(statistics.mode(
                    self.data[col]), inplace=True)
            elif self.data[col].dtype != object and self.data[col].isnull().sum() > 0:
                self.data[col].fillna(self.data[col].mean(), inplace=True)
        logging.info("Successfully Removed Missing Values in the data")

    def encoding_one_hot(self):
        self.data = pd.get_dummies(self.data, drop_first=True)
        self.columns = self.data.columns
        logging.info("Successfully Encoded Categorical Values in the data")

    def sampling(self, choice=False, size=1000):
        if choice:
            self.data = self.data.sample(n=size)
            logging.info("Successfully sampled the data")

    def remove(self, cols):
        self.data.drop(cols, axis=1, inplace=True)
        logging.info("Columns Removed Successfully")

    def scaling(self, choice=False):
        if choice:
            scaler = MinMaxScaler()
            self.data = pd.DataFrame(scaler.fit_transform(
                self.data), columns=self.columns)
            logging.info("Successfully scaled the data")
