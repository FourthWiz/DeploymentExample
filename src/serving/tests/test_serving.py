import pytest
from src.serving.run import app, Item
from fastapi.testclient import TestClient

client = TestClient(app)

@pytest.fixture
def data_negative():
    return {"age":39,
    "workclass": "Private",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"}

@pytest.fixture
def data_positive():
    return {"age":40,
    "workclass": "Private",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education-num": 17,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 10000,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"}

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}

def test_inference_negative(data_negative):
    response = client.post("/predict/", json=data_negative)
    print("response", response.json())
    assert response.status_code == 200
    assert response.json() == {
        "prediction": 0
    }

def test_inference_positive(data_positive):
    response = client.post("/predict/", json=data_positive)
    print("response", response.json())
    assert response.status_code == 200
    assert response.json() == {
        "prediction": 1
    }