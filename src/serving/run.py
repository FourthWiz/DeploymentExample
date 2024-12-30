#!/usr/bin/env python
"""
This script download a URL to a local destination
"""
import logging
import pandas as pd

import wandb
import pickle
from fastapi import FastAPI
from pydantic import BaseModel, Field
from enum import Enum
from hydra import initialize, compose
import os
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()



class Workclass(Enum):
    STATE_GOV = "State-gov"
    SE_NI = "Self-emp-not-inc"
    PRIVATE = "Private"
    FEDERAL_GOV = "Federal-gov"
    LOCAL_GOV = "Local-gov"
    UNKNOWN = "?"
    SE_INC = "Self-emp-inc"
    WITHOUT_PAY = "Without-pay"
    NEVER_WORKED = "Never-worked"


class Education(Enum):
    BACHELORS = "Bachelors"
    HS_GRAD = "HS-grad"
    ELEVENTH = "11th"
    MASTERS = "Masters"
    NINTH = "9th"
    SOME_COLLEGE = "Some-college"
    ASSOC_ACDM = "Assoc-acdm"
    ASSOC_VOC = "Assoc-voc"
    SEVENTH_EIGHTH = "7th-8th"
    DOCTORATE = "Doctorate"
    PROF_SCHOOL = "Prof-school"
    FIFTH_SIXTH = "5th-6th"
    TENTH = "10th"
    FIRST_FOURTH = "1st-4th"
    PRESCHOOL = "Preschool"
    TWELFTH = "12th"

class MariatalStatus(Enum):
    NEVER_MARRIED = "Never-married"
    MARRIED_CIV_SPOUSE = "Married-civ-spouse"
    DIVORCED = "Divorced"
    MARRIED_SPOUSE_ABSENT = "Married-spouse-absent"
    SEPARATED = "Separated"
    MARRIED_AF_SPOUSE = "Married-AF-spouse"
    WIDOWED = "Widowed"

class Occupation(Enum):
    ADM_CLERICAL = 'Adm-clerical'
    EXEC_MANAGERIAL = 'Exec-managerial'
    HANDLERS_CLEANERS = 'Handlers-cleaners'
    PROF_SPECIALTY = 'Prof-specialty'
    OTHER_SERVICE = 'Other-service'
    SALES = 'Sales'
    CRAFT_REPAIR = 'Craft-repair'
    TRANSPORT_MOVING = 'Transport-moving'
    FARMING_FISHING = 'Farming-fishing'
    MACHINE_OP_INSPCT = 'Machine-op-inspct'
    TECH_SUPPORT = 'Tech-support'
    UNKNOWN = '?'
    PROTECTIVE_SERV = 'Protective-serv'
    ARMED_FORCES = 'Armed-Forces'
    PRIV_HOUSE_SERV = 'Priv-house-serv'

class Relationship(Enum):
    NOT_IN_FAMILY = "Not-in-family"
    HUSBAND = "Husband"
    WIFE = "Wife"
    OWN_CHILD = "Own-child"
    UNMARRIED = "Unmarried"
    OTHER_RELATIVE = "Other-relative"

class Race(Enum):
    WHITE = "White"
    BLACK = "Black"
    ASIAN_PAC_ISLANDER = "Asian-Pac-Islander"
    AMER_INDIAN_ESKIMO = "Amer-Indian-Eskimo"
    OTHER = "Other"

class Sex(Enum):
    MALE = "Male"
    FEMALE = "Female"

class NativeCountry(Enum):
    UNITED_STATES = "United-States"
    CUBA = "Cuba"
    JAMAICA = "Jamaica"
    INDIA = "India"
    UNKNOWN = "?"
    MEXICO = "Mexico"
    SOUTH = "South"
    PUERTO_RICO = "Puerto-Rico"
    HONDURAS = "Honduras"
    ENGLAND = "England"
    CANADA = "Canada"
    GERMANY = "Germany"
    IRAN = "Iran"
    PHILIPPINES = "Philippines"
    ITALY = "Italy"
    POLAND = "Poland"
    COLUMBIA = "Columbia"
    CAMBODIA = "Cambodia"
    THAILAND = "Thailand"
    ECUADOR = "Ecuador"
    LAOS = "Laos"
    TAIWAN = "Taiwan"
    HAITI = "Haiti"
    PORTUGAL = 'Portugal'
    DOMINICAN_REPUBLIC = 'Dominican-Republic'
    EL_SALVADOR = 'El-Salvador'
    FRANCE = 'France'
    GUATEMALA = 'Guatemala'
    CHINA = 'China'
    JAPAN = 'Japan'
    YUGOSLAVIA = 'Yugoslavia'
    PERU = 'Peru'
    OUTLYING_US_GUAM_USVI_ETC = 'Outlying-US(Guam-USVI-etc)'
    SCOTLAND = 'Scotland'
    TRINIDAD_TOBAGO = 'Trinadad&Tobago'
    GREECE = 'Greece'
    NICARAGUA = 'Nicaragua'
    VIETNAM = 'Vietnam'
    HONGKONG = 'Hong'
    IRELAND = 'Ireland'
    HUNGARY = 'Hungary'
    HOLLAND_NETHERLANDS = 'Holand-Netherlands'


class Item(BaseModel):
    age: int = Field(example=39)
    workclass: Workclass
    fnlwgt: int = Field(example=77516, description="No clue what this is")
    education: Education
    education_num: int = Field(example=13, alias="education-num")
    marital_status: MariatalStatus = Field(alias="marital-status")
    occupation: Occupation
    relationship: Relationship
    race: Race
    sex: Sex
    capital_gain: int = Field(example=2174, alias="capital-gain")
    capital_loss: int = Field(example=0, alias="capital-loss")
    hours_per_week: int = Field(example=40, alias="hours-per-week")
    native_country: NativeCountry = Field(alias="native-country")

    def to_serializable(self):
        """
        Convert the Item instance to a JSON-serializable dictionary,
        handling Enums and other special types.
        """
        def serialize_value(value):
            if isinstance(value, Enum):
                return value.value
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [serialize_value(v) for v in value]
            elif isinstance(value, tuple):
                return tuple(serialize_value(v) for v in value)
            else:
                return value

        return {
            key: serialize_value(value)
            for key, value in self.dict(by_alias=True).items()
        }

app = FastAPI()

logging.info("Initializing Config")
with initialize(config_path=".", version_base="1.2"):
    config = compose(config_name="config.yaml")

wandb_api_key = os.environ["WANDB_API_KEY"]

if wandb_api_key:
    wandb.login(key=wandb_api_key)
    print("wandb initialized successfully.")
else:
    print("WANDB_API_KEY is not set. wandb will not be initialized.")

os.environ["WANDB_PROJECT"] = config["project_name"]
os.environ["WANDB_RUN_GROUP"] = config["experiment_name"]


logging.info("Initializing Weights & Biases")
run = wandb.init(job_type="inference")

logging.info("Downloading model %s", config.model_path)
address = run.use_artifact(config.model_path).download()
logging.info(f"Model downloaded to: {address}")
model = pickle.load(open(os.path.join(address, "model.pkl"), "rb"))


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict/")
def predict(item: Item):
    """
    Predicts the income of a person based on the input data.
    """
    logger.info(f"Received data: {item}")
    data = pd.DataFrame([item.to_serializable()])
    # for col in data.columns:
    #     data[col] = data[col].apply(lambda x: x.value if isinstance(x, Enum) else x)
    logging.info("Data: %s", data)
    y_pred = model.predict(data)[0]
    y_pred = int(y_pred) if isinstance(y_pred, (np.integer, np.int64)) else y_pred
    logging.info("Prediction: %s", y_pred)
    return {"prediction": y_pred}
