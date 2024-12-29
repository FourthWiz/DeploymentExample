import pytest
from src.modelling.utils import model
import pandas as pd

@pytest.fixture
def data():
    data = pd.read_csv("src/modelling/tests/test_sample.csv")
    return data

def test_model_init(data):
    model_lgb = model.ModelLGB(data)
    assert model_lgb.label_encoders is not None
    assert (not model_lgb.data.equals(data))

def test_model_train_first(data):
    model_lgb = model.ModelLGB(data)
    model_lgb.train_first()
    assert model_lgb.model is not None
    assert model_lgb.x_test is not None
    assert model_lgb.y_test is not None

def test_model_calc_metrics(data):
    model_lgb = model.ModelLGB(data)
    model_lgb.train_first()
    model_lgb.calc_metrics("pr_curve.png")
    assert model_lgb.roc_auc is not None