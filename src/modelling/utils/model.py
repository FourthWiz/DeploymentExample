import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, PrecisionRecallDisplay
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt

class ModelLGB:
    """
    Class for the LightGBM model.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Constructor for the LightGBM model.
        """
        self.data = data.copy()
        self.categorical_columns = [
                "native-country",
                "sex",
                "relationship",
                "occupation",
                "education",
                "workclass",
                "marital-status",
                "race"
            ]
        self.model = None
        self.roc_auc = None
        self.x_test = None
        self.y_test = None
        self.label_encoders = self._preprocess()


    def _preprocess(self):
        label_encoders = {}
        for col in self.categorical_columns:
            le = LabelEncoder()
            self.data.loc[:, col] = le.fit_transform(self.data[col])
            label_encoders[col] = le
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                self.data[col] = self.data[col].astype('int')
        return label_encoders
        

    def train_first(self):
        """
        Train the LightGBM model.
        """
        X = self.data.drop('income', axis=1)
        y = self.data['income']

        x_train, self.x_test, y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = lgb.LGBMClassifier()
        self.model.fit(x_train, y_train)

    
    def calc_metrics(self, plot_name):
        """
        Calculate the metrics for the LightGBM model.
        """
        y_pred_proba = self.model.predict_proba(self.x_test)[:, 1]

        self.roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        PrecisionRecallDisplay.from_estimator(self.model, self.x_test, self.y_test)
        plt.savefig(plot_name)


    
    def train_total(self):
        """
        Train the LightGBM model on the full dataset.
        """
        X = self.data.drop('income', axis=1)
        y = self.data['income']

        self.model = lgb.LGBMClassifier()
        self.model.fit(X, y)

    def predict(self, data: pd.DataFrame):
        """
        Make predictions using the LightGBM model.
        """
        for col in self.categorical_columns:
            data.loc[:, col] = self.label_encoders[col].transform(data[col])

        return self.model.predict(data)
    
    def sliced_predictions(self, slice_columns: list):
        """
        Make predictions using the LightGBM model.
        """
        results = {}
        for col in slice_columns:
            results[col] = self._slice_predict(self.x_test, self.y_test, col, self.label_encoders[col])

        return results
    
    def _slice_predict(self, x: pd.DataFrame, y, column:str, le: LabelEncoder):
        slices = x[column].unique()
        results = {}
        for slice in slices:
            predictions = self.model.predict_proba(x[x[column] == slice])[:, 1]
            slice_name = str(le.inverse_transform([slice])[
                0] if isinstance(slice, int) else slice)
            try:
                results[slice_name] = roc_auc_score(y[x[column] == slice], predictions)
            except:
                results[slice_name] = None
                continue
        return results
