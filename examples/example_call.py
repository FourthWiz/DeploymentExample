import requests

data_negative =  {"age":39,
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

response = requests.post("https://deploymentexample-411184494367.herokuapp.com/predict/", json=data_negative)

print("response:", response.json())
print("response status code:", response.status_code)