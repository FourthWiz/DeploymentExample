from ucimlrepo import fetch_ucirepo
import pandas as pd

def download():
    census_income = fetch_ucirepo(id=20) 

    X = census_income.data.features 
    y = census_income.data.targets 
    dataset = pd.concat([X, y], axis=1)
    return dataset