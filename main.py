import json
import logging

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig
from utils import move_extract

_steps = [
    "dataload", 
    "data_move_extract",
    "preprocessing"
]

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    with tempfile.TemporaryDirectory() as tmp_dir:

        if "dataload" in active_steps:
            logger.info("Downloading data")
            # Download file and load in W&B
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "dataload"),
                "main",
                env_manager="virtualenv",
                parameters={
                    "sample": config["dataload"]["filename"],
                    "artifact_name": "census.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )
    if "data_move_extract" in active_steps:
        logger.info("Moving data and extracting")
        # Move data to the correct location and extract it
        move_extract.move_extract(
            os.path.join(hydra.utils.get_original_cwd(), "src", "dataload", "data", config["dataload"]["filename"]),
            os.path.join(hydra.utils.get_original_cwd(), "data")
        )
    if "preprocessing" in active_steps:
        logger.info("Extracting and perprocessing data")

        _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "preprocessing"),
                "main",
                env_manager="virtualenv",
                parameters={
                    "filename": config["preprocessing"]["filename"],
                    "artifact_name": "census.csv",
                    "artifact_type": "data",
                    "artifact_description": "Preprocessed data"
                },
            )

            

if __name__ == "__main__":
    go()